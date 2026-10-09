# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import logging
import gzip
import time
import numpy as np
import torch
from tqdm import tqdm

# Your local modules
from layers import ConvGCNUNet
from regressors import UNetRegressor
import pgml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--weights", default="data/weights/gcn_n8/model.weights")
    parser.add_argument("--vcf", required=True, help="Path to input VCF")
    parser.add_argument("--ofile", default="predictions_output.bin", help="Output binary file")
    
    parser.add_argument("--min_log", default=1, type=int)
    parser.add_argument("--max_log", default=10, type=int)
    parser.add_argument("--n_scales", default=128, type=int)
    
    parser.add_argument("--embed_dim", default=32, type=int)
    parser.add_argument("--k", default=11, type=int)
    parser.add_argument("--wavelet", default="shannon")
    parser.add_argument("--wav_param", default=1, type=int)
    parser.add_argument("--dist", default="lorentz")
    
    # Target individuals and sequence length
    parser.add_argument("--target_ind", type=int, default=32)
    parser.add_argument("--L", type=float, default=23_000_000.0, help="Total length of the sequence in bp")
    
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)
        
    return args

def load_and_filter_vcf(vcf_path, target_ind=32, total_L=None):
    """Parses VCF, sets missing to 0.5, and drops uniform sites for the first target_ind individuals."""
    logging.info(f"Parsing VCF: {vcf_path}")
    
    open_func = gzip.open if vcf_path.endswith('.gz') else open
    pos_list = []
    geno_list = []
    
    with open_func(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
                
            cols = line.strip().split('\t')
            if len(cols) < 9:
                continue
                
            pos = float(cols[1])
            gts = cols[9:9+target_ind]
            
            has_ref = False
            has_alt = False
            row = []
            
            for gt_str in gts:
                gt_base = gt_str.split(':')[0]
                if '.' in gt_base:
                    row.append(0.5)
                elif '1' in gt_base:
                    row.append(1.0)
                    has_alt = True
                else:
                    row.append(0.0)
                    has_ref = True
                    
            # Filter uniform sites
            if has_ref and has_alt:
                pos_list.append(pos)
                geno_list.append(np.array(row, dtype=np.float32))
                
    pos_ndarray = np.array(pos_list, dtype=np.float32)
    
    # If positions are fractional [0, 1] (e.g. from msprime), scale them to physical bp!
    if total_L is not None and pos_ndarray[-1] <= 1.05:
        logging.info(f"Fractional positions detected. Scaling by L={total_L:g}")
        pos_ndarray *= total_L
        
    # Shape: (Individuals, SNPs)
    x_ndarray = np.stack(geno_list, axis=1) 
    
    logging.info(f"Kept {x_ndarray.shape[1]} polymorphic variants for {x_ndarray.shape[0]} individuals.")
    return x_ndarray, pos_ndarray

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using " + str(device) + " as device")
    
    # ---------------------------------------------------------
    # 1. Setup Model & Regressor
    # ---------------------------------------------------------
    weights_dir = '/'.join(args.weights.split('/')[:-1])
    pkl_path = os.path.join(weights_dir, 'train_val.pkl')
    stats_path = os.path.join(weights_dir, 'stats.npz')
    
    if os.path.exists(pkl_path):  
        _, _, _, (mu_x, std_x, mu_y, std_y) = pickle.load(open(pkl_path, 'rb'))
    elif os.path.exists(stats_path):
        ifile = np.load(stats_path)
        mu_x, std_x = ifile['mean_x'], ifile['std_x']
        mu_y, std_y = ifile['mean_y'], ifile['std_y']
    else:
        logging.warning("No stats found, using default 0,1 normalization")
        mu_x, std_x, mu_y, std_y = 0.0, 1.0, 0.0, 1.0
        
    scales = np.unique((2 ** np.linspace(args.min_log, args.max_log, args.n_scales)).astype(np.int32))
    model = ConvGCNUNet(in_dim=len(scales) + 2, embedding_dim=args.embed_dim, K=args.k, dist=args.dist).to(device)
    
    if args.wavelet == "shannon":
        config = pgml.WaveletConfig.shannon()
    elif args.wavelet == "gaussian":
        config = pgml.WaveletConfig.gaussian(order=int(args.wav_param))
    else:
        config = pgml.WaveletConfig.morlet(w0=args.wav_param)

    if args.weights != "None" and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    
    model.eval()
    
    # Pass to your regressor
    reg = UNetRegressor(model, scales, mu_x, std_x, mu_y, std_y)

    # ---------------------------------------------------------
    # 2. Parse VCF
    # ---------------------------------------------------------
    x_ndarray, pos_ndarray = load_and_filter_vcf(args.vcf, target_ind=args.target_ind, total_L=args.L)
    num_ind, num_snps = x_ndarray.shape
    
    # ---------------------------------------------------------
    # 3. Stream Inference over 1Mb Windows
    # ---------------------------------------------------------
    window_size_bp = 2_000_000.0
    current_bp = 0.0
    chunk_start_idx = 0
    total_base_pairs = float(pos_ndarray[-1]) if num_snps > 0 else 0.0
    
    MIN_SNPS = 1024 
    
    out_file = open(args.ofile, "wb")
    pbar = tqdm(total=int(total_base_pairs), unit="bp", desc="Streaming Predictions")

    with torch.no_grad():
        while chunk_start_idx < num_snps:
            target_bp = current_bp + window_size_bp
            
            chunk_end_idx = chunk_start_idx
            while chunk_end_idx < num_snps and pos_ndarray[chunk_end_idx] < target_bp:
                chunk_end_idx += 1
                
            if chunk_end_idx == chunk_start_idx:
                current_bp = target_bp
                pbar.update(int(window_size_bp))
                continue
                
            current_chunk_len = chunk_end_idx - chunk_start_idx
            
            # ---------------------------------------------------------
            # 4. Redundant Context Windowing & Local Position Scaling
            # ---------------------------------------------------------
            temp_start = chunk_start_idx
            temp_end = chunk_end_idx
            
            # Borrow context if the chunk is too small
            if temp_end - temp_start < MIN_SNPS:
                needed = MIN_SNPS - (temp_end - temp_start)
                temp_start = max(0, temp_start - needed)
                if temp_end - temp_start < MIN_SNPS:
                    needed = MIN_SNPS - (temp_end - temp_start)
                    temp_end = min(num_snps, temp_end + needed)
            
            x_chunk = x_ndarray[:, temp_start:temp_end]
            pos_chunk = pos_ndarray[temp_start:temp_end]
            
            # ---------------------------------------------------------
            # CRITICAL FIX: Localize positions to the CURRENT window
            # By subtracting current_bp, the network sees local coords [0, 1] 
            # instead of massive global coordinates like 23.5!
            # ---------------------------------------------------------
            pos_chunk_scaled = (pos_chunk - current_bp) / window_size_bp
            
            # Zero pad if the whole chromosome is still smaller than MIN_SNPS
            actual_len = temp_end - temp_start
            pad_len = 0
            if actual_len < MIN_SNPS:
                pad_len = MIN_SNPS - actual_len
                x_chunk = np.pad(x_chunk, ((0,0), (0, pad_len)), mode='constant')
                # Pad edges so the diffs don't artificially spike
                pos_chunk_scaled = np.pad(pos_chunk_scaled, (0, pad_len), mode='edge')
            
            # Predict using the full context
            D_pred_padded = reg.predict(x_chunk, pos_chunk_scaled)
            
            if pad_len > 0:
                D_pred_padded = D_pred_padded[:-pad_len, :]
                
            # SLICE MAGIC: Extract ONLY the segment belonging to our original target chunk
            offset = chunk_start_idx - temp_start
            D_pred = D_pred_padded[offset : offset + current_chunk_len, :]
            
            # Stream exactly identically to Rust: float32 raw bytes
            D_pred.astype(np.float32).tofile(out_file)
            
            pbar.update(int(window_size_bp))
            current_bp = target_bp
            chunk_start_idx = chunk_end_idx
            
    out_file.close()
    pbar.close()
    print(f"Predictions streamed to {args.ofile}")

if __name__ == "__main__":
    main()