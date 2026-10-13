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
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

import torch

# Your custom modeling modules
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
    
    # Rust Builder Argument
    parser.add_argument("--tsq_builder", default="./target/release/tsq_builder",
                        help="Path to the compiled Rust tsq_builder executable.")
    
    # Genetic Parameters (Fallbacks)
    parser.add_argument("--mu", default=1.5e-8, type=float, help="Default mutation rate.")
    parser.add_argument("--r", default=1.007e-8, type=float, help="Default recombination rate.")
    
    # Custom Model Arguments 
    parser.add_argument("-w", "--weights", default="models/n16_k11_cheby/best_model.pth", help="Path to PyTorch weights")
    
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

def summarize_results(csv_path: str, run_relate: bool, run_singer: bool):
    """Calculates and prints the relative percentage change in error metrics."""
    metrics = ['kc', 'rf', 'rms_log_coal', 'kl', 'chamfer', 'ntrees']
    custom_sums = {m: 0.0 for m in metrics}
    relate_sums = {m: 0.0 for m in metrics}
    singer_sums = {m: 0.0 for m in metrics}
    count = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            count += 1
            for m in metrics:
                # Accumulate custom metrics
                if row.get(f'custom_{m}') and row[f'custom_{m}'].strip():
                    # Handle arrays for some metrics (like kc, rf) by taking the mean of the array
                    # If it's a string representation of an array, safely parse it
                    val_str = row[f'custom_{m}']
                    if '[' in val_str:
                        val = np.mean(np.fromstring(val_str.strip('[]'), sep=' '))
                    else:
                        val = float(val_str)
                    custom_sums[m] += val
                
                # Accumulate relate metrics
                if run_relate and row.get(f'relate_{m}') and row[f'relate_{m}'].strip():
                    val_str = row[f'relate_{m}']
                    if '[' in val_str:
                        val = np.mean(np.fromstring(val_str.strip('[]'), sep=' '))
                    else:
                        val = float(val_str)
                    relate_sums[m] += val
                    
                # Accumulate singer metrics
                if run_singer and row.get(f'singer_{m}') and row[f'singer_{m}'].strip():
                    val_str = row[f'singer_{m}']
                    if '[' in val_str:
                        val = np.mean(np.fromstring(val_str.strip('[]'), sep=' '))
                    else:
                        val = float(val_str)
                    singer_sums[m] += val
                    
    if count == 0:
        return
        
    print("\n" + "="*80)
    print(f"BENCHMARK SUMMARY ({count} replicates)")
    print("="*80)
    
    # Header
    header = f"{'Metric':<20} | {'Custom vs. Relate':<20} | {'Custom vs. Singer':<20}"
    print(header)
    print("-" * len(header))
    
    for m in metrics:
        avg_custom = custom_sums[m] / count
        
        relate_str = "N/A"
        if run_relate:
            avg_relate = relate_sums[m] / count
            if avg_relate > 0:
                pct_diff = ((avg_custom - avg_relate) / avg_relate) * 100
                relate_str = f"{pct_diff:>+6.1f}%"
                
        singer_str = "N/A"
        if run_singer:
            avg_singer = singer_sums[m] / count
            if avg_singer > 0:
                pct_diff = ((avg_custom - avg_singer) / avg_singer) * 100
                singer_str = f"{pct_diff:>+6.1f}%"
                
        print(f"{m:<20} | {relate_str:<20} | {singer_str:<20}")
    
    print("="*80)
    print("Note: Negative percentages indicate Custom model has lower (better) error.\n")


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
            
            # Dynamically attach all training args to our current args namespace
            for key, value in parsed_args.items():
                setattr(args, key, value)
                
        # Force window_size to match L as specified in the JSON
        if hasattr(args, 'L'):
            args.window_size = args.L
            
        logging.info(f"Successfully loaded model hyper-parameters from {json_path}")
    else:
        logging.warning(f"arguments.json not found at {json_path}!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for custom model inference")
    
    # 2. Discover LMDBs
    lmdb_paths = sorted(find_lmdb_envs(args.input))
    if not lmdb_paths:
        print(f"No valid LMDB environments found at: {args.input}")
        return
    print(f"Found {len(lmdb_paths)} LMDB environment(s). Starting benchmark...")

    # 3. Setup Outputs
    csv_columns = [
        'db_name', 'key', 'N_est', 
        'relate_time', 'custom_time', 'singer_time',
        'relate_kc', 'custom_kc', 'singer_kc',
        'relate_rf', 'custom_rf', 'singer_rf',
        'relate_rms_log_coal', 'custom_rms_log_coal', 'singer_rms_log_coal',
        'relate_chamfer', 'custom_chamfer', 'singer_chamfer',
        'relate_ntrees', 'custom_ntrees', 'singer_ntrees',
        'relate_kl', 'custom_kl', 'singer_kl'
    ]
    
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for db_path in lmdb_paths:
            logging.info(f"Processing database: {db_path.name}")
            env = lmdb.open(str(db_path), readonly=True, lock=False)
            
            with env.begin() as txn:
                cursor = txn.cursor()
                
                # Retrieve keys and enforce n_samples limit
                keys = [key for key, _ in cursor]
                if args.n_samples is not None:
                    keys = keys[:args.n_samples]
                    
                # Setup tqdm progress bar for the keys in this DB
                for key in tqdm(keys, desc=f"Processing {db_path.name}"):
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

                    x = data['x'].astype(np.float32)
                    pos = data['pos']
                    params = data.get('params', {})
                    ts_raw = data['ts']
                    n_diploids_total = x.shape[0] // 2
                    
                    if hasattr(args, 'n') and args.n is not None and args.n < n_diploids_total:
                        paired_indices = np.array([[2*u, 2*u + 1] for u in range(n_diploids_total)], dtype=np.int32)
                        selected_pairs = paired_indices[:args.n]
                        ii = np.sort(selected_pairs.flatten())
                        
                        x = x[ii, :]
                        
                        is_ref = (x == 0)
                        is_alt = (x == 1)
                        valid_sites = np.where(is_ref.any(axis=0) & is_alt.any(axis=0))[0]
                        
                        x = x[:, valid_sites]
                        pos = pos[valid_sites]
                        
                    x = flip(x)
                    
                    # Build the evaluation PGTreeSequence
                    ts_true = PGTreeSequence.from_tskit(ts_raw, ii)
                    n_samples = x.shape[0]
                    
                    # Extract genetic parameters for the simulation
                    mu = params.get('mu', args.mu)
                    r = params.get('r', args.r)
                    sim_L = params.get('L', args.L) 
                    
                    N_est = watterson_estimate_N(S=len(pos), n=n_samples, mu=mu, L=sim_L)

                    # ---------------------------------------------------------
                    # 1. RUN INFERENCE METHODS
                    # ---------------------------------------------------------
                    relate_time, singer_time, pytorch_time = 0.0, 0.0, 0.0
                    
                    if run_relate:
                        start_time = time.time()
                        ts_est_relate = relate(x.astype(np.uint8), pos, n_samples, mu, r, sim_L, None, diploid=True, verbose=False)
                        relate_time = time.time() - start_time
                    
                    if run_singer:
                        start_time = time.time()
                        ts_est_singer = next(singer(x.astype(np.uint8), pos, mu, r, sim_L))
                        singer_time = time.time() - start_time
                    
                    start_time = time.time()
                    ts_est_custom = liana(args.weights, x, pos, args.L, args.window_size, device_str='cuda')
                    pytorch_time = time.time() - start_time
                    
                    # ---------------------------------------------------------
                    # 3. COMPUTE & SAVE METRICS
                    # ---------------------------------------------------------
                    row = {
                        'db_name': db_path.name,
                        'key': key_str,
                        'N_est': round(N_est, 2),
                        
                        'custom_time': round(pytorch_time, 4),
                        'custom_kc': ts_est_custom.kc_distance(ts_true),
                        'custom_rf': ts_est_custom.rf_distance(ts_true),
                        'custom_rms_log_coal': ts_est_custom.rms_log_coal_error(ts_true),
                        'custom_kl': ts_est_custom.symmetric_kl_divergence(ts_true),
                        'custom_chamfer': ts_est_custom.breakpoint_chamfer_distance(ts_true),
                        'custom_ntrees': np.abs(len(ts_est_custom.trees) - len(ts_true.trees)),
                    }
                    
                    if run_relate:
                        row.update({
                            'relate_time': round(relate_time, 4),
                            'relate_kc': ts_est_relate.kc_distance(ts_true),
                            'relate_rf': ts_est_relate.rf_distance(ts_true),
                            'relate_rms_log_coal': ts_est_relate.rms_log_coal_error(ts_true),
                            'relate_kl': ts_est_relate.symmetric_kl_divergence(ts_true),
                            'relate_chamfer': ts_est_relate.breakpoint_chamfer_distance(ts_true),
                            'relate_ntrees': np.abs(len(ts_est_relate.trees) - len(ts_true.trees)),
                        })
                        
                    if run_singer:
                        row.update({
                            'singer_time': round(singer_time, 4),
                            'singer_kc': ts_est_singer.kc_distance(ts_true),
                            'singer_rf': ts_est_singer.rf_distance(ts_true),
                            'singer_rms_log_coal': ts_est_singer.rms_log_coal_error(ts_true),
                            'singer_kl': ts_est_singer.symmetric_kl_divergence(ts_true),
                            'singer_chamfer': ts_est_singer.breakpoint_chamfer_distance(ts_true),
                            'singer_ntrees': np.abs(len(ts_est_singer.trees) - len(ts_true.trees)),
                        })
                        
                    writer.writerow(row)
                    
                    if args.verbose:
                        logging.info(f"[{key_str}] Custom: {pytorch_time:.2f}s" + 
                                     (f" | Relate: {relate_time:.2f}s" if run_relate else "") +
                                     (f" | Singer: {singer_time:.2f}s" if run_singer else ""))
                        
            env.close()
            
    # Calculate and display the percentage improvements
    summarize_results(args.output, run_relate, run_singer)
    print(f"\nBenchmarking complete. Results saved to {args.output}")

if __name__ == '__main__':
    main()