# -*- coding: utf-8 -*-
import os
import pickle
import logging
import numpy as np
import torch
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
import statistics
import scipy.stats as stats

# Your local modules
from liana.models.layers import ConvGCNUNet
from liana.models.regressors import UNetRegressor
import pgml
from popgenml.data.functions import distmat_to_tree 
from popgenml.data import PGTreeSequence

# =============================================================================
# Helper Functions (Must be at module level for multiprocessing serialization)
# =============================================================================

def build_tree_worker(d_mat, method):
    return distmat_to_tree(d_mat, method=method)[0]

def make_coherent(coals, trees):
    ret = dict()
    ret['tree'] = []
    ret['snps'] = []
    ret['coal'] = []
    
    ii = 0
    indices = [ii]
    T = trees[ii]
    _ = []
    ints = []
    
    start = 0
    
    while ii < len(coals) - 1:
        dist = trees[ii].kc_distance(trees[ii + 1])
        
        if dist > 0:
            ints.append((start, start + len(indices)))
            start += len(indices)
            
            ret['coal'].append(stats.hmean(np.array([coals[u] for u in indices]), axis = 0))
            ret['snps'].append(len(indices))
            ret['tree'].append(T)
        
            _.extend([ret['coal'][-1] for u in range(ret['snps'][-1])])
        
            indices = [ii + 1]
            dist = 0
        else:
            indices.append(ii + 1)

        ii += 1
        T = trees[ii]
        
    ret['tree'].append(T)
    ret['coal'].append(np.mean(np.array([coals[u] for u in indices]), axis = 0))
    ret['snps'].append(len(indices))
    ints.append((start, start + len(indices)))

    _.extend([ret['coal'][-1] for u in range(ret['snps'][-1])])
    
    return np.array(_), ret['snps'], ret['tree'], ints, np.array(ret['coal'])

def average_clade_times(tree_list):
    """
    Tracks clades across contiguous tskit trees, calculates the HARMONIC average 
    of their coalescent times, and returns a list of lists of averaged times.
    """
    runs = [] 
    active_runs = {} 
    
    for tree_idx, tree in enumerate(tree_list):
        current_clades = {}
        
        for u in tree.nodes():
            if not tree.is_sample(u):
                clade = frozenset(tree.samples(u))
                current_clades[clade] = tree.time(u)
                
        current_clade_set = set(current_clades.keys())
        active_clade_set = set(active_runs.keys())
        
        for clade in active_clade_set - current_clade_set:
            del active_runs[clade]
            
        for clade, time in current_clades.items():
            if clade not in active_runs:
                run_idx = len(runs)
                runs.append({'clade': clade, 'times': [], 'trees': []})
                active_runs[clade] = run_idx
            
            run_idx = active_runs[clade]
            runs[run_idx]['times'].append(time)
            runs[run_idx]['trees'].append(tree_idx)
            
    result = [[] for _ in range(len(tree_list))]
    
    for run in runs:
        avg_time = statistics.harmonic_mean(run['times'])
        for tree_idx in run['trees']:
            result[tree_idx].append(avg_time)
            
    for i in range(len(result)):
        result[i].sort()
        
    return result


# =============================================================================
# Main Standalone Inference Function
# =============================================================================

def liana(
    weights: str,
    x: np.ndarray,
    pos: np.ndarray,
    L: float = 2.5e6,
    window_size: float = 2.5e6,
    min_log: int = 1,
    max_log: int = 10,
    n_scales: int = 128,
    embed_dim: int = 32,
    k: int = 11,
    wavelet: str = "shannon",
    wav_param: int = 1,
    dist: str = "lorentz",
    method: str = "average",
    num_workers: int = 4,
    device_str: str = None
) -> PGTreeSequence:
    r"""
    Runs model inference on a sequence of variants, builds marginal trees in parallel, 
    post-processes coalescent times, and returns a PGTreeSequence.
    
    Args:
        weights (str): Path to the model weights file.
        x (np.ndarray): Binary genotype alignment array of shape (Individuals, SNPs).
        pos (np.ndarray): Position array scaled from 0 to 1 of length SNPs.
        L (float): Total length of the sequence in base pairs (used to scale pos).
        window_size (float): Size of the sliding window in base pairs.
        (Other args match the original argparse defaults).
        
    Returns:
        PGTreeSequence: Coherent marginal trees mapped to SNP intervals.
    """
    
    if device_str is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
        
    logging.info(f"Using {device} for inference")

    # 1. Setup Model & Regressor
    weights_dir = os.path.dirname(weights)
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
        
    scales = np.unique((2 ** np.linspace(min_log, max_log, n_scales)).astype(np.int32))
    model = ConvGCNUNet(in_dim=len(scales) + 2, embedding_dim=embed_dim, K=k, dist=dist).to(device)
    
    if wavelet == "shannon":
        config = pgml.WaveletConfig.shannon()
    elif wavelet == "gaussian":
        config = pgml.WaveletConfig.gaussian(order=int(wav_param))
    else:
        config = pgml.WaveletConfig.morlet(w0=wav_param)

    if weights != "None" and os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=device), strict=False)
    
    model.eval()
    reg = UNetRegressor(model, scales, mu_x, std_x, mu_y, std_y)

    # 2. Prepare Data Arrays
    x_ndarray = np.array(x, dtype=np.float32)
    pos_ndarray = np.array(pos, dtype=np.float32)
    num_ind, num_snps = x_ndarray.shape
    
    # Scale fractional [0,1] positions up to physical base pairs for the window logic
    if num_snps > 0 and pos_ndarray[-1] <= 1.05:
        pos_ndarray *= L
    
    # 3. Stream Inference over 1Mb Windows
    window_size_bp = window_size
    current_bp = 0.0
    chunk_start_idx = 0
    total_base_pairs = float(pos_ndarray[-1]) if num_snps > 0 else 0.0
    
    MIN_SNPS = 1024 
    trees_pred = []
    pbar = tqdm(total=int(total_base_pairs), unit="bp", desc="Processing Windows")

    tree_func = partial(build_tree_worker, method=method)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        with torch.no_grad():
            while chunk_start_idx < num_snps:
                target_bp = current_bp + window_size_bp
                
                chunk_end_idx = chunk_start_idx
                while chunk_end_idx < num_snps and pos_ndarray[chunk_end_idx] < target_bp:
                    chunk_end_idx += 1
                    
                if chunk_end_idx == chunk_start_idx:
                    current_bp = target_bp
                    pbar.update(int(window_size_bp))
                    continue
                    
                current_chunk_len = chunk_end_idx - chunk_start_idx
                
                # Context Windowing
                temp_start = chunk_start_idx
                temp_end = chunk_end_idx
                
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
                
                # Parallel Tree Generation
                chunk_trees = list(executor.map(tree_func, D_pred))
                trees_pred.extend(chunk_trees)
                
                pbar.update(int(window_size_bp))
                current_bp = target_bp
                chunk_start_idx = chunk_end_idx
                
    pbar.close()

    # 4. Post-processing Pipeline
    if not trees_pred:
        logging.warning("No trees were generated.")
        return PGTreeSequence(trees=[], snps=[])

    logging.info("Extracting coalescent times...")
    coal_times_pred = []
    for t in trees_pred:
        c = [t.time(u) for u in t.nodes()]
        c = sorted([u for u in c if u > 0])
        coal_times_pred.append(c)

    logging.info("Averaging clade times...")
    coal_times_pred_fwd = np.array(average_clade_times(trees_pred))
    coal_times_pred_rev = np.array(average_clade_times(trees_pred[::-1])[::-1])
    coal_times_pred_avg = (coal_times_pred_fwd + coal_times_pred_rev) / 2.

    logging.info("Making trees coherent...")
    coal_times_pred_final, snps_coherent, trees_coherent, intervals, coal = make_coherent(coal_times_pred_avg, trees_pred)
    
    logging.info("Building PGTreeSequence...")
    return PGTreeSequence(trees=trees_coherent, snps=snps_coherent)