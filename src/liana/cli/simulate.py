# -*- coding: utf-8 -*-
import os
import argparse
import logging

from popgenml.data.simulators import MSPrimeSimulator, DiscoalSimulator
from popgenml.data.functions import tree_to_distmat
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import itertools
import sys
import lmdb
import zlib

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--n_replicates", default = 1, type = int)
    parser.add_argument("--ifile", default = "None")

    parser.add_argument("--discoal", action = "store_true")

    parser.add_argument("--odir", default = "None")
    
    # <-- Added job_id to prevent key collisions on a cluster
    parser.add_argument("--job_id", default = 0, type = int, help="Global job ID for LMDB keys")
    parser.add_argument("--eval", action = "store_true")
    
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.debug('root: made output directory {0}'.format(args.odir))
    # ${odir_del_block}

    return args

def main():
    args = parse_args()

    if not args.discoal:
        sim = MSPrimeSimulator(args.ifile)
    else:
        sim = DiscoalSimulator(args.ifile)
    # number of haploids
    n = 2 * sim.samples['pop1']['n']

    db_path = os.path.join(args.odir, f'simulations_{args.job_id}.lmdb')
    env = lmdb.open(db_path, map_size=1099511627776)
    
    for ix in range(args.n_replicates):
        ret = sim.simulate(verbose = True)
        
        ts = ret['ts']
        
        if type(ts) != list:
            ts = ts.simplify(reduce_to_site_topology = True)
            tree = ts.first()
        else:
            ij = 0
            tree = ts[ij].first()
            
        print('have genotype matrix shape: {}'.format(ret['x'].shape))
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
        
        Ds = np.array(Ds, dtype = np.float32)
        print('have {} trees...'.format(Ds.shape[0]))
        sys.stdout.flush()
        
        data_dict = {
            'x': ret['x'].astype(np.uint8),
            'intervals': np.array(intervals),
            'pos': ret['pos'],
            'coal': np.array(coal_times, dtype = np.float32),
            'D': Ds,
            'params': sim.params,
        }
        if args.eval:
            data_dict['ts'] = ts
        
        byte_data = pickle.dumps(data_dict, protocol=-1)
        
        compressed_data = zlib.compress(byte_data, level=3)
        
        key = f"{args.job_id:08d}_{ix:08d}".encode('ascii')
        
        with env.begin(write=True) as txn:
            txn.put(key, compressed_data)
            
    # Close the environment once the loop is finished
    env.close()

if __name__ == '__main__':
    main()
