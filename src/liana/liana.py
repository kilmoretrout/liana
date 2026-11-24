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

def extract_canonical_clades(tree):
    """
    Extracts canonical topology strings matching liana/lib/src/main.rs:
    - Leaves are represented as `str(u)`.
    - Internal nodes are represented as `({child0},{child1})` with children sorted lexicographically.

    Returns:
        tuple: (topology_hash, clades_dict) where clades_dict maps canonical clade string -> node time.
    """
    topologies = {}
    clades = {}
    for u in tree.nodes(order="postorder"):
        if tree.is_sample(u):
            topologies[u] = str(u)
        else:
            children = sorted(topologies[c] for c in tree.children(u))
            topo = f"({','.join(children)})"
            topologies[u] = topo
            clades[topo] = float(tree.time(u))
    return topologies[tree.root], clades

def apply_smoothed_times(tree, smoothed_times, eps=1e-7):
    """
    Rebuilds a tskit.Tree with internal node times replaced by `smoothed_times`
    in ascending internal node ID (UPGMA merge) order, matching main.rs:build_newick.
    Applies a minimal monotonic nudge only if needed to satisfy tskit's strict parent > child invariant.
    """
    tables = tree.tree_sequence.dump_tables()
    internal_nodes = sorted([u for u in tree.nodes() if not tree.is_sample(u)])

    times = np.asarray(smoothed_times, dtype=np.float64).copy()
    for i in range(len(times)):
        if i == 0:
            if times[i] < eps:
                times[i] = eps
        else:
            min_allowed = max(times[i - 1] + eps, np.nextafter(times[i - 1], np.inf))
            if times[i] < min_allowed:
                times[i] = min_allowed

    for idx, node_id in enumerate(internal_nodes):
        tables.nodes[node_id] = tables.nodes[node_id].replace(time=float(times[idx]))

    tables.sort()
    return tables.tree_sequence().first(sample_lists=True)

def make_coherent(coals, trees, topo_hashes=None):
    """
    Collapses contiguous runs of identical tree topologies (by canonical topology hash,
    matching main.rs) and rebuilds each region's tree with the smoothed coalescent times
    at `start_idx`.
    """
    if topo_hashes is None:
        topo_hashes = [extract_canonical_clades(t)[0] for t in trees]

    ret_trees = []
    ret_snps = []
    ret_coals = []
    ints = []
    expanded_coals = []

    current_topology = topo_hashes[0]
    start_idx = 0

    for i in range(1, len(trees)):
        if topo_hashes[i] != current_topology:
            region_len = i - start_idx
            region_coal = coals[start_idx]
            ints.append((start_idx, i))
            ret_snps.append(region_len)
            ret_coals.append(region_coal)
            ret_trees.append(apply_smoothed_times(trees[start_idx], region_coal))
            expanded_coals.extend([region_coal] * region_len)

            current_topology = topo_hashes[i]
            start_idx = i

    region_len = len(trees) - start_idx
    region_coal = coals[start_idx]
    ints.append((start_idx, len(trees)))
    ret_snps.append(region_len)
    ret_coals.append(region_coal)
    ret_trees.append(apply_smoothed_times(trees[start_idx], region_coal))
    expanded_coals.extend([region_coal] * region_len)

    return np.array(expanded_coals), ret_snps, ret_trees, ints, np.array(ret_coals)

def average_clade_times(tree_list):
    """
    Tracks canonical-topology clades across contiguous trees (matching main.rs:average_clade_times),
    calculates the harmonic average of their coalescent times, and returns sorted smoothed times per tree.

    Accepts either a sequence of tskit.Tree objects or pre-extracted clade dicts {clade_str: time}.
    """
    runs = []
    active_runs = {}

    for tree_idx, item in enumerate(tree_list):
        if isinstance(item, dict):
            current_clades = item
        else:
            _, current_clades = extract_canonical_clades(item)

        current_clade_set = set(current_clades.keys())
        active_clade_set = set(active_runs.keys())

        for clade in active_clade_set - current_clade_set:
            del active_runs[clade]

        for clade, time in current_clades.items():
            if clade not in active_runs:
                run_idx = len(runs)
                runs.append(([], []))
                active_runs[clade] = run_idx

            run_idx = active_runs[clade]
            runs[run_idx][0].append(time)
            runs[run_idx][1].append(tree_idx)

    result = [[] for _ in range(len(tree_list))]

    for times, tree_indices in runs:
        has_zero = False
        sum_inv = 0.0
        for t in times:
            if t <= 0.0:
                has_zero = True
                break
            sum_inv += 1.0 / t

        if has_zero or sum_inv == 0.0:
            avg_time = 0.0
        else:
            avg_time = len(times) / sum_inv

        for tree_idx in tree_indices:
            result[tree_idx].append(avg_time)

    for res in result:
        res.sort()

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
    device_str: str = None,
    verbose: bool = False
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
        
    if verbose:
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
    reg = UNetRegressor(model, scales, mu_x, std_x, mu_y, std_y, device=device)

    # 2. Prepare Data Arrays
    x_ndarray = np.array(x, dtype=np.float32)
    pos_ndarray = np.array(pos, dtype=np.float64)
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
    if verbose:
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
                    if verbose:
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
                if verbose:
                    pbar.update(int(window_size_bp))
                current_bp = target_bp
                chunk_start_idx = chunk_end_idx
    if verbose:
        pbar.close()
        logging.info("Extracting coalescent times...")

    # 4. Post-processing Pipeline
    if not trees_pred:
        logging.warning("No trees were generated.")
        return PGTreeSequence(trees=[], snps=[])

    extracted = [extract_canonical_clades(t) for t in trees_pred]
    topo_hashes = [h for h, _ in extracted]
    clades_list = [c for _, c in extracted]

    coal_times_pred_fwd = np.array(average_clade_times(clades_list), dtype=np.float64)
    coal_times_pred_rev = np.array(average_clade_times(clades_list[::-1])[::-1], dtype=np.float64)
    coal_times_pred_avg = (coal_times_pred_fwd + coal_times_pred_rev) / 2.0

    coal_times_pred_final, snps_coherent, trees_coherent, intervals, coal = make_coherent(
        coal_times_pred_avg, trees_pred, topo_hashes=topo_hashes
    )
    
    return PGTreeSequence(trees=trees_coherent, snps=snps_coherent)