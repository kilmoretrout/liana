# -*- coding: utf-8 -*-
import argparse
import os
import time
import csv
import pickle
import lmdb
import numpy as np
from pathlib import Path

from popgenml.data import PGTreeSequence, relate

def watterson_estimate_N(S, n, mu, L):
    """
    Computes Watterson's estimate for the diploid effective population size (N).
    
    S: number of segregating sites (SNPs)
    n: number of sampled haploids
    mu: per-site, per-generation mutation rate
    L: sequence length
    """
    if S == 0 or n <= 1:
        return 10000.0  # Fallback if no SNPs are present to avoid division by zero
        
    # Watterson's correction factor (a_n)
    a_n = sum(1.0 / i for i in range(1, n))
    
    # theta_W = S / a_n
    # Since theta = 4 * N * mu * L (for diploids)
    # N = theta_W / (4 * mu * L)
    N_est = S / (4.0 * mu * L * a_n)
    return N_est

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Relate on LMDB databases using Watterson's N.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a single LMDB or a directory containing multiple LMDBs.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the benchmark results CSV.")
    
    # Fallback parameters
    parser.add_argument("--mu", default=1.5e-8, type=float, help="Default mutation rate.")
    parser.add_argument("--r", default=1.007e-8, type=float, help="Default recombination rate.")
    parser.add_argument("--L", default=1000000, type=float, help="Default sequence length.")
    parser.add_argument("--verbose", action="store_true", help="Print inference progress.")
    
    return parser.parse_args()

def find_lmdb_envs(target_path):
    """
    Identifies whether the target is a single LMDB or a directory of LMDBs 
    and returns a list of valid LMDB paths.
    """
    target = Path(target_path)
    
    if target.suffix == '.lmdb' or (target.is_dir() and (target / "data.mdb").exists()):
        return [target]
    
    if target.is_dir():
        envs = list(target.rglob("*.lmdb")) + [p.parent for p in target.rglob("data.mdb")]
        return list(set(envs))
    
    return []

def main():
    args = parse_args()
    
    lmdb_paths = find_lmdb_envs(args.input)
    if not lmdb_paths:
        print(f"No valid LMDB environments found at: {args.input}")
        return

    print(f"Found {len(lmdb_paths)} LMDB environment(s). Starting benchmark...")

    # Set up CSV Output
    csv_columns = [
        'db_name', 'key', 'inference_time_s', 'estimated_N',
        'kc_err', 'rf_err', 'rms_log_coal_err', 'chamfer_err'
    ]
    
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for db_path in lmdb_paths:
            print(f"Processing database: {db_path.name}")
            
            env = lmdb.open(str(db_path), readonly=True, lock=False)
            
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    key_str = key.decode('ascii')
                    
                    try:
                        data = pickle.loads(value)
                    except Exception as e:
                        print(f"  [Error] Failed to unpickle key {key_str}: {e}")
                        continue
                    
                    if 'ts' not in data:
                        print(f"  [Skip] Key {key_str} missing 'ts' object. Update simulator to save 'ts'.")
                        continue

                    # Extract tensors
                    x = data['x']
                    pos = data['pos']
                    params = data.get('params', {})
                    
                    # Map inference parameters 
                    mu = params.get('mu', args.mu)
                    r = params.get('r', args.r)
                    L = params.get('L', args.L)
                    
                    # 1. Estimate N using Watterson's estimator
                    n_samples = x.shape[0]
                    S_snps = len(pos)
                    N_est = watterson_estimate_N(S=S_snps, n=n_samples, mu=mu, L=L)

                    # 2. Run Inference
                    start_time = time.time()
                    ts_est = relate(
                        x=x, 
                        pos=pos, 
                        sample_size=n_samples, 
                        mu=mu, 
                        r=r, 
                        L=L, 
                        N=N_est, 
                        verbose=args.verbose
                    )
                    inference_time = time.time() - start_time
                    
                    # 3. Reconstruct true TreeSequence
                    ts_true = PGTreeSequence.from_tskit(data['ts'])

                    # 4. Compute Metrics
                    try:
                        kc_err = ts_est.average_kc_distance(ts_true)
                        rf_err = ts_est.average_rf_distance(ts_true)
                        rms_coal_err = ts_est.average_rms_log_coal_time(ts_true)
                        chamfer_err = ts_est.breakpoint_chamfer_distance(ts_true)
                    except Exception as e:
                        print(f"  [Error] Metric computation failed for {key_str}: {e}")
                        continue

                    # 5. Save results
                    writer.writerow({
                        'db_name': db_path.name,
                        'key': key_str,
                        'inference_time_s': round(inference_time, 4),
                        'estimated_N': round(N_est, 2),
                        'kc_err': kc_err,
                        'rf_err': rf_err,
                        'rms_log_coal_err': rms_coal_err,
                        'chamfer_err': chamfer_err
                    })
                    
                    if args.verbose:
                        print(f"  Processed {key_str} in {inference_time:.2f}s | N ≈ {N_est:.0f} | "
                              f"KC: {kc_err:.2f} | RF: {rf_err:.2f}")

            env.close()
            
    print(f"\nBenchmarking complete. Results saved to {args.output}")

if __name__ == '__main__':
    main()
