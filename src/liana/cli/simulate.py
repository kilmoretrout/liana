# -*- coding: utf-8 -*-
import os
import argparse
import logging
import json
import pickle
import sys
import itertools
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

from popgenml.data.simulators import MSPrimeSimulator, DiscoalSimulator
from popgenml.data.functions import tree_to_distmat

# ${imports}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--n_replicates", default=1, type=int)
    parser.add_argument("--ifile", default="None")
    parser.add_argument("--discoal", action="store_true")
    parser.add_argument("--odir", default="None")
    parser.add_argument("--job_id", default=0, type=int, help="Global job ID")
    parser.add_argument("--eval", action="store_true")
    
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.makedirs(args.odir, exist_ok=True)
            logging.debug(f'root: made output directory {args.odir}')

    return args

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

def main():
    args = parse_args()

    if not args.discoal:
        sim = MSPrimeSimulator(args.ifile)
    else:
        sim = DiscoalSimulator(args.ifile)
        
    n = 2 * sim.samples['pop1']['n']

    # 1. Define the Arrow Schema
    schema = pa.schema([
        ('key', pa.string()),
        ('x', pa.list_(pa.uint8())),
        ('x_shape', pa.list_(pa.int32())),
        ('intervals', pa.list_(pa.float64())),
        ('intervals_shape', pa.list_(pa.int32())),
        ('pos', pa.list_(pa.float64())),
        ('coal', pa.list_(pa.float32())),
        ('coal_shape', pa.list_(pa.int32())),
        ('D', pa.list_(pa.float32())),
        ('D_shape', pa.list_(pa.int32())),
        ('params', pa.string()),
        ('ts_bytes', pa.binary()) # Nullable by default
    ])

    parquet_path = os.path.join(args.odir, f'simulations_{args.job_id}.parquet')
    
    # 2. Open ParquetWriter to stream directly to disk
    with pq.ParquetWriter(parquet_path, schema, compression='snappy') as writer:
        for ix in range(args.n_replicates):
            ret = sim.simulate(verbose=True)
            ts = ret['ts']
            
            if type(ts) != list:
                ts = ts.simplify(reduce_to_site_topology=True)
                tree = ts.first()
            else:
                ij = 0
                tree = ts[ij].first()
                
            print(f"have genotype matrix shape: {ret['x'].shape}")
            sys.stdout.flush()
            
            Ds = []
            coal_times = []
            
            if type(ts) != list:
                intervals = []
            else:
                intervals = ret['intervals']
                
            while True:
                D = tree_to_distmat(tree)
                coal_times.append(sorted([tree.time(u) for u in tree.nodes() if not tree.is_leaf(u)]))
                Ds.append(D)
                
                if type(ts) != list:
                    intervals.append(tree.interval)
                    if not tree.next():
                        break
                else:
                    ij += 1
                    if ij < len(ts):
                        tree = ts[ij].first()
                    else:
                        break
            
            Ds = np.array(Ds, dtype=np.float32)
            coal_times = np.array(coal_times, dtype=np.float32)
            intervals = np.array(intervals, dtype=np.float64)
            x_mat = ret['x'].astype(np.uint8)
            pos_arr = ret['pos'].astype(np.float64)
            
            print(f"have {Ds.shape[0]} trees...")
            sys.stdout.flush()
            
            # 3. Serialize parameters and eval objects
            params_str = json.dumps(sim.params, cls=NumpyEncoder)
            ts_bytes = pickle.dumps(ts, protocol=-1) if args.eval else None
            
            # 4. Construct a single row as an Arrow RecordBatch
            # We flatten multi-dimensional arrays and store their shapes
            batch = pa.RecordBatch.from_arrays([
                pa.array([f"{args.job_id:08d}_{ix:08d}"]),
                pa.array([x_mat.ravel()]),
                pa.array([list(x_mat.shape)]),
                pa.array([intervals.ravel()]),
                pa.array([list(intervals.shape)]),
                pa.array([pos_arr]),
                pa.array([coal_times.ravel()]),
                pa.array([list(coal_times.shape)]),
                pa.array([Ds.ravel()]),
                pa.array([list(Ds.shape)]),
                pa.array([params_str]),
                pa.array([ts_bytes])
            ], schema=schema)
            
            # 5. Stream batch to file
            writer.write_batch(batch)

if __name__ == '__main__':
    main()