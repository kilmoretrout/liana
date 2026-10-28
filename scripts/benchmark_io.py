# -*- coding: utf-8 -*-
import time
import argparse
import lmdb
import zlib
import pickle
import pyarrow.parquet as pq
import numpy as np

def benchmark_lmdb(lmdb_path, n_reads):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    # Grab the first N keys
    with env.begin() as txn:
        keys = [key for key, _ in txn.cursor().iternext(keys=True, values=False)]
    keys = keys[:n_reads]
    
    start_time = time.time()
    
    with env.begin() as txn:
        for k in keys:
            # 1. Disk Read
            byte_data = txn.get(k)
            # 2. Decompress (CPU bound)
            raw_bytes = zlib.decompress(byte_data)
            # 3. Deserialize (CPU bound / GIL lock)
            data = pickle.loads(raw_bytes)
            
            # Instantiate arrays (simulating dataloader payload)
            x = data['x']
            pos = data['pos']
            D = data['D']
            
    end_time = time.time()
    env.close()
    
    elapsed = end_time - start_time
    throughput = len(keys) / elapsed
    print(f"[LMDB] Read {len(keys)} replicates in {elapsed:.3f}s ({throughput:.1f} reps/sec)")

def benchmark_parquet(parquet_path, n_reads):
    pf = pq.ParquetFile(parquet_path)
    reads = min(n_reads, pf.num_row_groups)
    
    start_time = time.time()
    
    for i in range(reads):
        # 1. Disk Read (reads only this specific row group)
        rg = pf.read_row_group(i)
        
        # 2. Zero-copy C++ to NumPy mapping (Bypasses Python completely)
        # .values.to_numpy() hands the underlying C++ memory pointer directly to NumPy
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
    print(f"[Parquet] Read {reads} replicates in {elapsed:.3f}s ({throughput:.1f} reps/sec)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", type=str, required=True, help="Path to .lmdb folder")
    parser.add_argument("--parquet", type=str, required=True, help="Path to .parquet file")
    parser.add_argument("--n", type=int, default=1000, help="Number of replicates to read")
    args = parser.parse_args()
    
    print(f"Benchmarking {args.n} reads...")
    benchmark_lmdb(args.lmdb, args.n)
    benchmark_parquet(args.parquet, args.n)