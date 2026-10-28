# -*- coding: utf-8 -*-
import os
import glob
import time
import argparse
import lmdb
import zlib
import pickle
import pyarrow.parquet as pq
import numpy as np

def benchmark_lmdb_folder(folder_path, n_reads):
    files = glob.glob(os.path.join(folder_path, "*.lmdb"))
    if not files:
        print(f"[LMDB] No .lmdb files found in {folder_path}")
        return

    # 1. Setup: Gather items across all files
    items = []
    for f in files:
        try:
            env = lmdb.open(f, readonly=True, lock=False)
            with env.begin() as txn:
                for key in txn.cursor().iternext(keys=True, values=False):
                    items.append((f, key))
                    if len(items) >= n_reads:
                        break
            env.close()
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")
        if len(items) >= n_reads:
            break

    reads = len(items)
    print(f"[LMDB] Found {reads} replicates across {len(set(x[0] for x in items))} files. Starting read test...")

    # 2. Open environments (mimics dataloader worker caching)
    envs = {f: lmdb.open(f, readonly=True, lock=False, readahead=False, meminit=False) 
            for f in set([x[0] for x in items])}

    # 3. Timed Loop
    start_time = time.time()
    for f, k in items:
        with envs[f].begin() as txn:
            byte_data = txn.get(k)
        
        # Decompress and unpickle
        try:
            raw_bytes = zlib.decompress(byte_data)
        except zlib.error:
            raw_bytes = byte_data
            
        data = pickle.loads(raw_bytes)
        
        # Instantiate arrays to RAM
        x = data['x']
        pos = data['pos']
        D = data['D']

    end_time = time.time()
    
    # 4. Cleanup
    for env in envs.values():
        env.close()

    elapsed = end_time - start_time
    throughput = reads / elapsed
    print(f"[LMDB] Read {reads} replicates in {elapsed:.3f}s ({throughput:.1f} reps/sec)\n")


def benchmark_parquet_folder(folder_path, n_reads):
    files = glob.glob(os.path.join(folder_path, "*.parquet"))
    if not files:
        print(f"[Parquet] No .parquet files found in {folder_path}")
        return

    # 1. Setup: Gather row groups across all files
    items = []
    for f in files:
        try:
            pf = pq.ParquetFile(f)
            for i in range(pf.num_row_groups):
                items.append((f, i))
                if len(items) >= n_reads:
                    break
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")
        if len(items) >= n_reads:
            break

    reads = len(items)
    print(f"[Parquet] Found {reads} replicates across {len(set(x[0] for x in items))} files. Starting read test...")

    # 2. Open ParquetFile handles (mimics dataloader worker caching)
    pfs = {f: pq.ParquetFile(f) for f in set([x[0] for x in items])}

    # 3. Timed Loop
    start_time = time.time()
    for f, rg_idx in items:
        rg = pfs[f].read_row_group(rg_idx)
        
        # Zero-copy C++ to NumPy mapping
        x_flat = rg.column('x')[0].values.to_numpy()
        x_shape = rg.column('x_shape')[0].values.to_numpy()
        x = x_flat.reshape(x_shape)
        
        pos = rg.column('pos')[0].values.to_numpy()
        
        D_flat = rg.column('D')[0].values.to_numpy()
        D_shape = rg.column('D_shape')[0].values.to_numpy()
        D = D_flat.reshape(D_shape)

    end_time = time.time()
    
    elapsed = end_time - start_time
    throughput = reads / elapsed
    print(f"[Parquet] Read {reads} replicates in {elapsed:.3f}s ({throughput:.1f} reps/sec)\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_dir", type=str, required=True, help="Directory containing .lmdb files/folders")
    parser.add_argument("--parquet_dir", type=str, required=True, help="Directory containing .parquet files")
    parser.add_argument("--n", type=int, default=1000, help="Number of replicates to read")
    args = parser.parse_args()
    
    print(f"--- I/O Benchmark: {args.n} Replicates ---")
    benchmark_lmdb_folder(args.lmdb_dir, args.n)
    benchmark_parquet_folder(args.parquet_dir, args.n)
