# -*- coding: utf-8 -*-
import argparse
import os
import time
import csv
import pickle
import lmdb
import numpy as np
import tempfile
import subprocess
import io
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import tskit
from Bio import Phylo

# Your custom modeling modules
from layers import ConvGCNUNet
from regressors import UNetRegressor
import pgml

# Tree seq tools
from popgenml.data import PGTreeSequence, relate

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Relate vs Custom GCN-UNet + Rust Builder on LMDB simulations.")
    
    # I/O Arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a single LMDB or a directory containing multiple LMDBs.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the benchmark results CSV.")
    parser.add_argument("--verbose", action="store_true", help="Print inference progress.")
    
    # Rust Builder Argument
    parser.add_argument("--tsq_builder", default="./target/release/tsq_builder",
                        help="Path to the compiled Rust tsq_builder executable.")
    
    # Genetic Parameters (Fallbacks)
    parser.add_argument("--mu", default=1.5e-8, type=float, help="Default mutation rate.")
    parser.add_argument("--r", default=1.007e-8, type=float, help="Default recombination rate.")
    parser.add_argument("--L", default=1000000.0, type=float, help="Default sequence length.")
    
    # Custom Model Arguments
    parser.add_argument("--weights", default="data/weights/gcn_n8/model.weights", help="Path to PyTorch weights")
    parser.add_argument("--min_log", default=1, type=int)
    parser.add_argument("--max_log", default=10, type=int)
    parser.add_argument("--n_scales", default=128, type=int)
    parser.add_argument("--embed_dim", default=32, type=int)
    parser.add_argument("--k", default=11, type=int)
    parser.add_argument("--wavelet", default="shannon")
    parser.add_argument("--wav_param", default=1, type=int)
    parser.add_argument("--dist", default="lorentz")
    parser.add_argument("--window_size", default=2e6, type=float, help="Stream chunk size in bp")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
    return args

def watterson_estimate_N(S, n, mu, L):
    """Computes Watterson's estimate for the diploid effective population size (N)."""
    if S == 0 or n <= 1:
        return 10000.0
    a_n = sum(1.0 / i for i in range(1, n))
    return S / (4.0 * mu * L * a_n)

def find_lmdb_envs(target_path):
    target = Path(target_path)
    if target.suffix == '.lmdb' or (target.is_dir() and (target / "data.mdb").exists()):
        return [target]
    if target.is_dir():
        envs = list(target.rglob("*.lmdb")) + [p.parent for p in target.rglob("data.mdb")]
        return list(set(envs))
    return []

def write_temp_vcf(x_ndarray, pos_ndarray, vcf_path):
    """
    Writes the LMDB genotype and position matrices into a temporary VCF format 
    so the Rust tsq_builder can parse it.
    """
    num_ind, num_snps = x_ndarray.shape
    with open(vcf_path, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        header += [f"ind_{i}" for i in range(num_ind)]
        f.write("\t".join(header) + "\n")
        
        for j in range(num_snps):
            pos = int(pos_ndarray[j])
            row = ["1", str(pos), ".", "A", "T", ".", "PASS", ".", "GT"]
            for i in range(num_ind):
                val = x_ndarray[i, j]
                if val == 1.0: gt = "1"
                elif val == 0.0: gt = "0"
                else: gt = "."
                row.append(gt)
            f.write("\t".join(row) + "\n")

def load_custom_model(args, device):
    """Initializes the UNetRegressor and loads weights/normalization stats."""
    logging.info(f"Loading custom model weights from {args.weights}")
    weights_dir = os.path.dirname(args.weights)
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
    
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    else:
        logging.warning(f"Weights file not found at {args.weights}. Using uninitialized weights!")
        
    model.eval()
    return UNetRegressor(model, scales, mu_x, std_x, mu_y, std_y)

def run_custom_inference(reg, x_ndarray, pos_ndarray, window_size_bp, out_bin_path):
    """Streams PyTorch inference over chunked windows and writes raw float32s to a binary file."""
    num_ind, num_snps = x_ndarray.shape
    current_bp = 0.0
    chunk_start_idx = 0
    MIN_SNPS = 1024 
    
    with open(out_bin_path, "wb") as out_file, torch.no_grad():
        while chunk_start_idx < num_snps:
            target_bp = current_bp + window_size_bp
            
            chunk_end_idx = chunk_start_idx
            while chunk_end_idx < num_snps and pos_ndarray[chunk_end_idx] < target_bp:
                chunk_end_idx += 1
                
            if chunk_end_idx == chunk_start_idx:
                current_bp = target_bp
                continue
                
            current_chunk_len = chunk_end_idx - chunk_start_idx
            temp_start, temp_end = chunk_start_idx, chunk_end_idx
            
            if temp_end - temp_start < MIN_SNPS:
                needed = MIN_SNPS - (temp_end - temp_start)
                temp_start = max(0, temp_start - needed)
                if temp_end - temp_start < MIN_SNPS:
                    needed = MIN_SNPS - (temp_end - temp_start)
                    temp_end = min(num_snps, temp_end + needed)
            
            x_chunk = x_ndarray[:, temp_start:temp_end]
            pos_chunk = pos_ndarray[temp_start:temp_end]
            
            pos_chunk_scaled = (pos_chunk - current_bp) / window_size_bp
            
            actual_len = temp_end - temp_start
            pad_len = 0
            if actual_len < MIN_SNPS:
                pad_len = MIN_SNPS - actual_len
                x_chunk = np.pad(x_chunk, ((0,0), (0, pad_len)), mode='constant')
                pos_chunk_scaled = np.pad(pos_chunk_scaled, (0, pad_len), mode='edge')
            
            D_pred_padded = reg.predict(x_chunk, pos_chunk_scaled)
            
            if pad_len > 0:
                D_pred_padded = D_pred_padded[:-pad_len, :]
                
            offset = chunk_start_idx - temp_start
            D_pred = D_pred_padded[offset : offset + current_chunk_len, :]
            
            D_pred.astype(np.float32).tofile(out_file)
            
            current_bp = target_bp
            chunk_start_idx = chunk_end_idx

def load_tsv_to_treesequence(tsv_path: str, L: float) -> PGTreeSequence:
    """
    Parses the Rust TSV containing Newick strings into a PGTreeSequence object
    using Biopython and tskit.
    """
    trees = []
    intervals = []
    
    with open(tsv_path, 'r') as f:
        header = next(f)  # Skip the header
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
                
            start_bp = float(parts[0])
            end_bp = float(parts[1])
            nwk_str = parts[2]
            
            # Scale intervals to [0.0, 1.0]
            left = start_bp / L
            right = end_bp / L
            intervals.append((left, right))
            
            span = max(right - left, 1e-12)
            
            # Parse Newick string into a tskit.Tree
            b_tree = Phylo.read(io.StringIO(nwk_str), "newick")
            tables = tskit.TableCollection(sequence_length=span)
            
            clade_to_time = {}
            for clade in b_tree.find_clades(order="postorder"):
                if clade.is_terminal():
                    clade_to_time[clade] = 0.0
                else:
                    c_times = []
                    for c in clade.clades:
                        child_time = clade_to_time[c]
                        b_len = c.branch_length or 0.0
                        if b_len <= 0: 
                            b_len = 1e-9 
                        c_times.append(child_time + b_len)
                    clade_to_time[clade] = max(c_times)
                    
            clade_to_node = {}
            leaf_map = {}
            for clade in b_tree.get_terminals():
                try:
                    leaf_id = int(clade.name)
                    leaf_map[leaf_id] = clade
                except (ValueError, TypeError):
                    pass
                    
            if not leaf_map:
                raise ValueError("Newick leaves must have integer names.")
                
            max_leaf_id = max(leaf_map.keys())
            
            for i in range(max_leaf_id + 1):
                if i in leaf_map:
                    clade = leaf_map[i]
                    node_id = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=clade_to_time[clade])
                    clade_to_node[clade] = node_id
                else:
                    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)

            for clade in b_tree.find_clades(order="postorder"):
                if not clade.is_terminal():
                    time_val = clade_to_time[clade]
                    node_id = tables.nodes.add_row(flags=0, time=time_val)
                    clade_to_node[clade] = node_id

            for clade in b_tree.find_clades():
                if clade in clade_to_node:
                    parent_id = clade_to_node[clade]
                    for child in clade.clades:
                        child_id = clade_to_node[child]
                        tables.edges.add_row(
                            left=0.0, 
                            right=span, 
                            parent=parent_id, 
                            child=child_id
                        )

            tables.sort()
            ts = tables.tree_sequence()
            trees.append(ts.first())
            
    return PGTreeSequence(trees=trees, intervals=intervals)

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for custom model inference")
    
    if not os.path.exists(args.tsq_builder):
        logging.error(f"Rust executable not found at {args.tsq_builder}. Check path.")
        return

    # 1. Initialize Custom Model globally
    reg = load_custom_model(args, device)
    
    # 2. Discover LMDBs
    lmdb_paths = find_lmdb_envs(args.input)
    if not lmdb_paths:
        print(f"No valid LMDB environments found at: {args.input}")
        return
    print(f"Found {len(lmdb_paths)} LMDB environment(s). Starting benchmark...")

    # 3. Setup Outputs
    csv_columns = [
        'db_name', 'key', 'N_est', 
        'relate_time_s', 'custom_pytorch_time_s', 'custom_rust_time_s',
        'relate_kc', 'custom_kc', 
        'relate_rf', 'custom_rf', 
        'relate_rms_log_coal', 'custom_rms_log_coal', 
        'relate_chamfer', 'custom_chamfer',
        'relate_ntrees', 'custrom_ntrees',
        'relate_kl', 'custom_kl'
    ]
    
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for db_path in lmdb_paths:
            logging.info(f"Processing database: {db_path.name}")
            env = lmdb.open(str(db_path), readonly=True, lock=False)
            
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    key_str = key.decode('ascii')
                    
                    try:
                        data = pickle.loads(value)
                    except Exception as e:
                        logging.error(f"Failed to unpickle key {key_str}: {e}")
                        continue
                    
                    if 'ts' not in data:
                        logging.warning(f"Key {key_str} missing 'ts' object. Skipping.")
                        continue

                    # Extract tensors from LMDB
                    x = data['x'].astype(np.float32)
                    pos = data['pos']
                    
                    
                    
                    params = data.get('params', {})
                    ts_true = PGTreeSequence.from_tskit(data['ts'])
                    
                    # Parameters
                    mu = params.get('mu', args.mu)
                    r = params.get('r', args.r)
                    L = params.get('L', args.L)
                    n_samples = x.shape[0]
                    N_est = watterson_estimate_N(S=len(pos), n=n_samples, mu=mu, L=L)

                    # ---------------------------------------------------------
                    # 1. RUN RELATE
                    # ---------------------------------------------------------
                    start_time = time.time()
                    ts_est_relate = relate(x, pos, n_samples, mu, r, L, N_est, verbose=False)
                    relate_time = time.time() - start_time
                    
                    # ---------------------------------------------------------
                    # 2. RUN CUSTOM PIPELINE (PyTorch -> Rust TSQ Builder)
                    # ---------------------------------------------------------
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_vcf = os.path.join(tmpdir, "temp.vcf")
                        tmp_bin = os.path.join(tmpdir, "temp.bin")
                        tmp_tsv = os.path.join(tmpdir, "temp.tsv")
                        
                        # Step A: Write temp VCF for Rust tool
                        write_temp_vcf(x, pos, tmp_vcf)
                        
                        # Step B: PyTorch Inference -> write BIN
                        pt_start = time.time()
                        run_custom_inference(reg, x, pos, args.window_size, tmp_bin)
                        pytorch_time = time.time() - pt_start
                        
                        # Step C: Rust tsq_builder -> write TSV
                        rust_start = time.time()
                        rust_cmd = [
                            args.tsq_builder,
                            "--vcf", tmp_vcf,
                            "--bin", tmp_bin,
                            "--output", tmp_tsv
                        ]
                        
                        try:
                            subprocess.run(rust_cmd, check=True, capture_output=True)
                        except subprocess.CalledProcessError as e:
                            logging.error(f"Rust tsq_builder failed on key {key_str}:\n{e.stderr.decode()}")
                            continue
                            
                        rust_time = time.time() - rust_start
                        
                        # Step D: Parse TSV into PGTreeSequence
                        try:
                            ts_est_custom = load_tsv_to_treesequence(tmp_tsv, L=L)
                        except Exception as e:
                            logging.error(f"Failed parsing TSV for key {key_str}: {e}")
                            continue

                    # ---------------------------------------------------------
                    # 3. COMPUTE & SAVE METRICS
                    # ---------------------------------------------------------
                    try:
                        writer.writerow({
                            'db_name': db_path.name,
                            'key': key_str,
                            'N_est': round(N_est, 2),
                            
                            'relate_time_s': round(relate_time, 4),
                            'custom_pytorch_time_s': round(pytorch_time, 4),
                            'custom_rust_time_s': round(rust_time, 4),
                            
                            'relate_kc': ts_est_relate.average_kc_distance(ts_true),
                            'custom_kc': ts_est_custom.average_kc_distance(ts_true),
                            
                            'relate_rf': ts_est_relate.average_rf_distance(ts_true),
                            'custom_rf': ts_est_custom.average_rf_distance(ts_true),
                            
                            'relate_rms_log_coal': ts_est_relate.average_rms_log_coal_time(ts_true),
                            'custom_rms_log_coal': ts_est_custom.average_rms_log_coal_time(ts_true),
                            
                            'relate_chamfer': ts_est_relate.breakpoint_chamfer_distance(ts_true),
                            'custom_chamfer': ts_est_custom.breakpoint_chamfer_distance(ts_true),
                            
                            'relate_ntrees': np.abs(len(ts_est_relate.trees) - len(ts_true.trees)),
                            'custom_ntrees': np.abs(len(ts_est_custom.trees) - len(ts_true.trees)),
                            
                            'relate_kl': ts_est_relate.average_symmetric_kl_divergence(ts_true),
                            'custom_kl': ts_est_custom.average_symmetric_kl_divergence(ts_true),
                        })
                        if args.verbose:
                            logging.info(f"[{key_str}] Relate: {relate_time:.2f}s | "
                                         f"Custom(PT+Rust): {pytorch_time+rust_time:.2f}s")
                            
                    except Exception as e:
                        logging.error(f"Metric computation failed for {key_str}: {e}")
                        continue

            env.close()
            
    print(f"\nBenchmarking complete. Results saved to {args.output}")

if __name__ == '__main__':
    main()