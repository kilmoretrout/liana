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
    
    # Target individuals for the benchmark
    parser.add_argument("--target_ind", type=int, default=32)
    
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)
        
    return args

def load_and_filter_vcf(vcf_path, target_ind=32):
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
            
            row = []
            for gt_str in gts:
                gt_base = gt_str.split(':')[0]
                if '.' in gt_base:
                    row.append(0.5)
                elif '1' in gt_base:
                    row.append(1.0)
                else:
                    row.append(0.0)
                    
            row_arr = np.array(row, dtype=np.float32)
            
            # Filter uniform sites
            if not np.all(row_arr == row_arr[0]):
                pos_list.append(pos)
                geno_list.append(row_arr)
                
    pos_ndarray = np.array(pos_list, dtype=np.float32)
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
    x_ndarray, pos_ndarray = load_and_filter_vcf(args.vcf, target_ind=args.target_ind)
    num_ind, num_snps = x_ndarray.shape
    
    # ---------------------------------------------------------
    # 3. Stream Inference over 1Mb Windows
    # ---------------------------------------------------------
    window_size_bp = 1_000_000.0
    current_bp = 0.0
    chunk_start_idx = 0
    total_base_pairs = float(pos_ndarray[-1]) if num_snps > 0 else 0.0
    
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
                pbar.update(window_size_bp)
                continue
                
            # Extract current chunk
            x_chunk = x_ndarray[:, chunk_start_idx:chunk_end_idx]
            pos_chunk = pos_ndarray[chunk_start_idx:chunk_end_idx]
            
            # Divide positions by 1,000,000 so the regressor's internal 
            # `np.diff(pos)` calculation perfectly matches the Rust normalization!
            pos_chunk_scaled = pos_chunk / window_size_bp
            
            # Predict using your class
            t0 = time.time()
            D_pred = reg.predict(x_chunk, pos_chunk_scaled)
            
            # Stream exactly identically to Rust: float32 raw bytes
            D_pred.astype(np.float32).tofile(out_file)
            
            # Update loop variables and UI
            pbar.update(int(window_size_bp))
            current_bp = target_bp
            chunk_start_idx = chunk_end_idx
            
    out_file.close()
    pbar.close()
    print(f"Predictions streamed to {args.ofile}")

if __name__ == "__main__":
    main()
