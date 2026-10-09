# -*- coding: utf-8 -*-
import os
import argparse
import logging

from popgenml.data.simulators import MSPrimeSimulator, DiscoalSimulator
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

def tree_to_dist_mat(tree):
    tree = tree.split_polytomies()

    leaf_nodes = sorted([u for u in tree.nodes() if tree.is_leaf(u)])
    n = len(leaf_nodes)
    
    D = np.zeros((n * (n - 1) // 2,))

    for ix, (i, j) in enumerate(itertools.combinations(leaf_nodes, 2)):
        D[ix] = tree.distance_between(i, j)

    return D

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--n_replicates", default = 1, type = int)
    parser.add_argument("--ifile", default = "None")

    parser.add_argument("--simple", action = "store_true")
    parser.add_argument("--write_vcf", action = "store_true")
    parser.add_argument("--discoal", action = "store_true")

    parser.add_argument("--odir", default = "None")
    
    # <-- Added job_id to prevent key collisions on a cluster
    parser.add_argument("--job_id", default = 0, type = int, help="Global job ID for LMDB keys") 
    
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

def rescale_times(coal_times):
    # n - 1
    dcoal = np.diff(coal_times, prepend = 0.)
    n = coal_times.shape[0] + 1
    
    n = np.array(range(n, 1, -1))
    nC2 = n * (n - 1) / 2.    
    
    dcoal *= nC2
    
    return np.cumsum(dcoal)

def main():
    args = parse_args()

    if not args.discoal:
        sim = MSPrimeSimulator(args.ifile)
    else:
        sim = DiscoalSimulator(args.ifile)
    # number of haploids
    n = 2 * sim.samples['pop1']['n']
    
    # --- Open LMDB Environment ---
    # map_size is the virtual memory limit (1TB here). It will be a sparse file on Linux.
    db_path = os.path.join(args.odir, 'simulations.lmdb')
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
        
        if args.write_vcf:
            with open(os.path.join(args.odir, '{0:04d}.vcf'.format(ix)), "w") as f:
                ts.write_vcf(f)
            pickle.dump(ts, open(os.path.join(args.odir, '{0:04d}.pkl'.format(ix)), 'wb'))
            
        print('have genotype matrix shape: {}'.format(ret['x'].shape))
        sys.stdout.flush()
        
        Ds = []
        coal_times = []
        
        if type(ts) != list:
            intervals = []
        else:
            intervals = ret['intervals']
            
        while True:
            D = tree_to_dist_mat(tree)
                  
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
        
        Ds = np.array(Ds)
        print('have {} trees...'.format(Ds.shape[0]))
        sys.stdout.flush()
        
        # --- Package and Write to LMDB ---
        # 1. Package all the arrays into a single dictionary
        data_dict = {
            'x': ret['x'].astype(np.uint8),
            'intervals': np.array(intervals),
            'pos': ret['pos'],
            'coal': np.array(coal_times),
            'D': Ds,
            'params': sim.params
        }
        
        # 2. Serialize to bytes using fastest C-pickler protocol available
        byte_data = pickle.dumps(data_dict, protocol=-1)
        
        # 3. Create a unique key combining job_id and replicate index
        # Format: "00000012_00000450" (Job 12, Replicate 450)
        key = f"{args.job_id:08d}_{ix:08d}".encode('ascii')
        
        # 4. Write to the database
        with env.begin(write=True) as txn:
            txn.put(key, byte_data)
            
    # Close the environment once the loop is finished
    env.close()

if __name__ == '__main__':
    main()
