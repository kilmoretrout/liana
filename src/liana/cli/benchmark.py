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
import json
import gc
import psutil
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

import torch

# Your liana modeling modules
from liana.models.layers import ConvGCNUNet
from liana.models.regressors import UNetRegressor
import pgml
import zlib

# Tree seq tools
from popgenml.data import PGTreeSequence, relate, singer
from popgenml.data.functions import flip

from liana import liana

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Relate vs Custom GCN-UNet + Rust Builder on LMDB simulations.")
    
    # I/O Arguments
    parser.add_argument("-i", "--input", type=str, required=True,
                    help="Path to a single LMDB or a directory containing multiple LMDBs.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to save the benchmark results CSV.")
    parser.add_argument("--verbose", action="store_true", help="Print inference progress.")
    
    # Benchmarking Targets
    parser.add_argument("-b", "--bench_against", choices=['all', 'singer', 'relate'], default='all',
                        help="Which baseline tools to benchmark against Liana. Default is 'all'.")
    
    # Samples Argument
    parser.add_argument("-n", "--n_samples", type=int, default=None,
                        help="Number of replicates/chunks to process per database. Defaults to all.")
    
    parser.add_argument("--n", default = 2, type = int, help = "number of diploid pairs to evaluate on")
    
    # Rust Builder Argument
    parser.add_argument("--tsq_builder", default="./target/release/tsq_builder",
                        help="Path to the compiled Rust tsq_builder executable.")
    
    # Genetic Parameters (Fallbacks)
    parser.add_argument("--mu", default=1.5e-8, type=float, help="Default mutation rate.")
    parser.add_argument("--r", default=1.007e-8, type=float, help="Default recombination rate.")
    
    # Custom Model Arguments 
    parser.add_argument("-w", "--weights", default="models/n16_k11_cheby/best_model.pth", help="Path to PyTorch weights")
    parser.add_argument("--two_pop", action = "store_true", help = "For two-pop data.  Assumes the same number of haploids in each population when downsampling to 2*n haploids")
    
    args = parser.parse_args()
    
    # Force logging to INFO so memory logs always show, even without --verbose
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
        
    return args

def log_mem(step_name):
    """Helper function to log current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    logging.info(f"[MEMORY] {step_name}: {mem_mb:.2f} MB")

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

def summarize_results(csv_path: str, run_relate: bool, run_singer: bool):
    # [Unchanged]
    pass

def main():
    args = parse_args()
    
    # Establish which baselines to run
    run_relate = args.bench_against in ['all', 'relate']
    run_singer = args.bench_against in ['all', 'singer']
    
    # ---------------------------------------------------------
    # Load Model Hyperparameters from JSON
    # ---------------------------------------------------------
    weights_path = Path(args.weights)
    json_path = weights_path.parent / "arguments.json"
    
    if json_path.exists():
        with open(json_path, "r") as f:
            config = json.load(f)
            parsed_args = config.get("parsed_args", {})
            
            for key, value in parsed_args.items():
                if not (key in ["weights", "n"]):
                    setattr(args, key, value)
                
        if hasattr(args, 'L'):
            args.window_size = args.L
            
        logging.info(f"Successfully loaded model hyper-parameters from {json_path}")
    else:
        logging.warning(f"arguments.json not found at {json_path}!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for liana model inference")
    
    #log_mem("After initial setup")

    # 2. Discover LMDBs
    lmdb_paths = sorted(find_lmdb_envs(args.input))
    if not lmdb_paths:
        print(f"No valid LMDB environments found at: {args.input}")
        return
    print(f"Found {len(lmdb_paths)} LMDB environment(s). Starting benchmark...")

    # 3. Setup Outputs
    csv_columns = [
        'db_name', 'key', 'N_est', 
        'relate_time', 'liana_time', 'singer_time',
        'relate_kc', 'liana_kc', 'singer_kc',
        'relate_rf', 'liana_rf', 'singer_rf',
        'relate_rms_log_coal', 'liana_rms_log_coal', 'singer_rms_log_coal',
        'relate_chamfer', 'liana_chamfer', 'singer_chamfer',
        'relate_ntrees', 'liana_ntrees', 'singer_ntrees',
        'relate_kl', 'liana_kl', 'singer_kl'
    ]
    
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for db_path in tqdm(lmdb_paths, desc=f"Processing databases"):
            env = lmdb.open(str(db_path), readonly=True, lock=False)
            
            with env.begin() as txn:
                cursor = txn.cursor()
                keys = [key for key, _ in cursor]
                if args.n_samples is not None:
                    keys = keys[:args.n_samples]
                    
                #log_mem(f"Before DB loop ({len(keys)} keys loaded)")

                for key in keys:
                    log_mem(f"Start of loop (Key: {key.decode('ascii')})")
                    value = txn.get(key)
                    key_str = key.decode('ascii')
                    
                    try:
                        try:
                            raw_bytes = zlib.decompress(value)
                        except zlib.error:
                            raw_bytes = value
                        data = pickle.loads(raw_bytes)
                    except Exception as e:
                        logging.error(f"Failed to unpickle key {key_str}: {e}")
                        continue
                        
                    #log_mem(f"After unpickling data (Key: {key_str})")

                    x = data['x'].astype(np.float32)
                    pos = data['pos']
                    params = data.get('params', {})
                    ts_raw = data['ts']
                    n_diploids_total = x.shape[0] // 2
                    
                    if hasattr(args, 'n') and args.n is not None and args.n < n_diploids_total:
                        paired_indices = np.array([[2*u, 2*u + 1] for u in range(n_diploids_total)], dtype=np.int32)
                        n = args.n
                        total_pairs = x.shape[0] // 2
                        
                        if args.two_pop:
                            half_n = n // 2
                            mid = total_pairs // 2
                            first_half = paired_indices[:mid]
                            last_half = paired_indices[mid:]
                            selected_pairs = np.vstack((first_half[:half_n], last_half[:half_n]))
                        else:
                            selected_pairs = paired_indices[:n]
       
                        ii = np.sort(selected_pairs.flatten())
                        x = x[ii, :]
                        
                        is_ref = (x == 0)
                        is_alt = (x == 1)
                        valid_sites = np.where(is_ref.any(axis=0) & is_alt.any(axis=0))[0]
                        
                        x = x[:, valid_sites]
                        pos = pos[valid_sites]
                        
                    x = flip(x)
                                        
                    ts_true = PGTreeSequence.from_tskit(ts_raw, ii)
                    n_samples = x.shape[0]
                    
                    mu = params.get('mu', args.mu)
                    r = params.get('r', args.r)
                    sim_L = params.get('L', args.L) 
                    
                    N_est = watterson_estimate_N(S=len(pos), n=n_samples, mu=mu, L=sim_L)

                    relate_time, singer_time, pytorch_time = 0.0, 0.0, 0.0
                    
                    if run_relate:
                        start_time = time.time()
                        ts_est_relate = relate(x.astype(np.uint8), pos, n_samples, mu, r, sim_L, None, diploid=True, verbose=False)
                        relate_time = time.time() - start_time
                        #log_mem(f"After Relate (Key: {key_str})")
                    
                    if run_singer:
                        start_time = time.time()
                        ts_est_singer = next(singer(x.astype(np.uint8), pos, mu, r, sim_L))
                        singer_time = time.time() - start_time
                        #log_mem(f"After Singer (Key: {key_str})")
                    
                    start_time = time.time()
                    ts_est_liana = liana(args.weights, x, pos, args.L, args.window_size, device_str='cuda')
                    pytorch_time = time.time() - start_time
                    #log_mem(f"After Liana (Key: {key_str})")
                    
                    row = {
                        'db_name': db_path.name,
                        'key': key_str,
                        'N_est': round(N_est, 2),
                        
                        'liana_time': round(pytorch_time, 4),
                        'liana_kc': ts_est_liana.kc_distance(ts_true),
                        'liana_rf': ts_est_liana.rf_distance(ts_true),
                        'liana_rms_log_coal': ts_est_liana.rms_log_coal_error(ts_true),
                        'liana_kl': ts_est_liana.symmetric_kl_divergence(ts_true),
                        'liana_chamfer': ts_est_liana.breakpoint_chamfer_distance(ts_true),
                        'liana_ntrees': np.abs(ts_est_liana.num_trees() - ts_true.num_trees()),
                    }
                    
                    if run_relate:
                        row.update({
                            'relate_time': round(relate_time, 4),
                            'relate_kc': ts_est_relate.kc_distance(ts_true),
                            'relate_rf': ts_est_relate.rf_distance(ts_true),
                            'relate_rms_log_coal': ts_est_relate.rms_log_coal_error(ts_true),
                            'relate_kl': ts_est_relate.symmetric_kl_divergence(ts_true),
                            'relate_chamfer': ts_est_relate.breakpoint_chamfer_distance(ts_true),
                            'relate_ntrees': np.abs(ts_est_relate.num_trees() - ts_true.num_trees()),
                        })
                        
                    if run_singer:
                        row.update({
                            'singer_time': round(singer_time, 4),
                            'singer_kc': ts_est_singer.kc_distance(ts_true),
                            'singer_rf': ts_est_singer.rf_distance(ts_true),
                            'singer_rms_log_coal': ts_est_singer.rms_log_coal_error(ts_true),
                            'singer_kl': ts_est_singer.symmetric_kl_divergence(ts_true),
                            'singer_chamfer': ts_est_singer.breakpoint_chamfer_distance(ts_true),
                            'singer_ntrees': np.abs(ts_est_singer.num_trees() - ts_true.num_trees()),
                        })
                        
                    writer.writerow(row)
                    
                    # Cleanup variables to prevent memory leaks in the loop
                    del data, x, pos, ts_true
                    if run_relate: del ts_est_relate
                    if run_singer: del ts_est_singer
                    del ts_est_liana
                    
                    # Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    #log_mem(f"End of loop after cleanup (Key: {key_str})")
            env.close()
            
    # summarize_results(args.output, run_relate, run_singer)
    print(f"\nBenchmarking complete. Results saved to {args.output}")

if __name__ == '__main__':
    main()