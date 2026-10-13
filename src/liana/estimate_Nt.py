import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ete3 import Tree
from pathlib import Path
import gzip
from collections import Counter

def get_coalescent_times(newick_str):
    """
    Parses a Newick string and returns a sorted list of internal node ages.
    """
    t = Tree(newick_str)
    times = []
    
    for node in t.traverse("postorder"):
        if node.is_leaf():
            node.age = 0.0
        else:
            child = node.children[0]
            node.age = child.age + child.dist
            times.append(node.age)
            
    return sorted(times)

def calc_skyline_components(sorted_times, time_bins):
    """
    Calculates the coalescent events and lineage opportunities within each time bin.
    """
    n_leaves = len(sorted_times) + 1
    n_bins = len(time_bins) - 1
    
    events = np.zeros(n_bins)
    opportunities = np.zeros(n_bins)
    
    current_lineages = n_leaves
    event_idx = 0
    
    for i in range(n_bins):
        t_start = time_bins[i]
        t_end = time_bins[i+1]
        t_current = t_start
        
        while event_idx < len(sorted_times) and sorted_times[event_idx] <= t_end:
            t_event = sorted_times[event_idx]
            
            if t_event > t_current:
                dt = t_event - t_current
                opportunities[i] += dt * (current_lineages * (current_lineages - 1)) / 2.0
                
            events[i] += 1
            current_lineages -= 1
            t_current = t_event
            event_idx += 1
            
        if t_current < t_end and current_lineages > 1:
            dt = t_end - t_current
            opportunities[i] += dt * (current_lineages * (current_lineages - 1)) / 2.0
            
    return events, opportunities

def update_expected_sfs(newick_str, weight, exp_sfs):
    """
    Calculates the expected folded SFS contribution for a single tree based on branch lengths.
    """
    t = Tree(newick_str)
    n_leaves = len(t)
    
    for node in t.traverse("postorder"):
        if node.is_leaf():
            node.leaf_count = 1
        else:
            node.leaf_count = sum(c.leaf_count for c in node.children)
            
        if node.is_root():
            continue
            
        # Folded frequency index
        k = min(node.leaf_count, n_leaves - node.leaf_count)
        if k > 0:
            exp_sfs[k] += node.dist * weight

def parse_vcf_sfs(vcf_path, obs_sfs):
    """
    Parses a VCF file, filters monomorphic/masked sites, and updates the observed folded SFS.
    """
    open_func = gzip.open if str(vcf_path).endswith('.gz') else open
    
    with open_func(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
                
            cols = line.strip().split('\t')
            c0, c1 = 0, 0
            
            # Parse genotypes (matching the Rust logic: 0.5 for '.', 1.0 for '1', 0.0 for '0')
            for i in range(9, 9 + 32):
                gt_base = cols[i].split(':')[0]
                if '.' in gt_base:
                    continue
                elif '1' in gt_base:
                    c1 += 1
                else:
                    c0 += 1
                    
            # Filter sites with one or less unique non-masked values
            if c0 > 0 and c1 > 0:
                k = min(c0, c1)
                obs_sfs[k] += 1

def main():
    parser = argparse.ArgumentParser(description="Estimate N(t) and compare SFS from Tree Sequence TSVs & VCFs")
    parser.add_argument("-i", "--input", required=True, help="Path to a single .tsv or a directory containing .tsv files")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of trees to sample per TSV for demographic history and SFS")
    parser.add_argument("--n_bins", type=int, default=35, help="Number of time bins (omit to auto-determine)")
    parser.add_argument("--output", default="demographic_sfs_panel.png", help="Output plot filename")
    args = parser.parse_args()

    input_path = Path(args.input)
    tsv_files = []
    
    if input_path.is_file():
        tsv_files = [input_path]
    elif input_path.is_dir():
        tsv_files = list(input_path.glob("*.tsv"))
        if not tsv_files:
            raise ValueError(f"No .tsv files found in directory: {input_path}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    print(f"Found {len(tsv_files)} TSV file(s) to process.")
    
    all_trees_times = []
    total_exp_sfs = Counter()
    total_obs_sfs = Counter()
    
    for tsv_file in tsv_files:
        print(f"\nProcessing: {tsv_file}")
        df = pd.read_csv(tsv_file, sep='\t')
        
        # 1. Sample trees weighted by physical distance
        lengths = df['End_BP'] - df['Start_BP']
        weights = lengths / lengths.sum()
        
        print(f"  Sampling {args.n_samples} trees based on sequence length...")
        sampled_indices = np.random.choice(df.index, size=args.n_samples, p=weights, replace=True)
        sampled_newicks = df.loc[sampled_indices, 'Newick'].values
        
        # 2. Extract coalescent times and compute expected SFS from the SAMPLED trees
        print("  Extracting coalescent times and Expected SFS...")
        for newick in sampled_newicks:
            # Skyline times
            times = get_coalescent_times(newick)
            all_trees_times.append(times)
            
            # Expected SFS
            # Since we sampled proportional to physical distance, every sampled tree
            # holds equal statistical weight. We just pass weight=1.0.
            update_expected_sfs(newick, 1.0, total_exp_sfs)
            
        # 3. Observed SFS from corresponding VCF
        vcf_file = tsv_file.with_suffix('.vcf.gz')
        if not vcf_file.exists():
            vcf_file = tsv_file.with_suffix('.vcf') # Try unzipped
            
        if vcf_file.exists():
            print(f"  Computing Observed SFS from {vcf_file.name}...")
            parse_vcf_sfs(vcf_file, total_obs_sfs)
        else:
            print(f"  [!] Warning: Could not find corresponding VCF ({vcf_file.name}) for SFS calculation.")

    # --- Skyline Demographic Calculation ---
    print("\nCalculating Skyline opportunities...")
    flat_times = np.concatenate(all_trees_times)
    flat_times = flat_times[flat_times > 1e-6] 
    
    target_bins = args.n_bins if args.n_bins is not None else max(15, min(int(2 * (len(flat_times) ** (1/3))), 150))
    time_bins = np.logspace(np.log10(np.min(flat_times) * 0.95), np.log10(np.max(flat_times) * 1.05), target_bins + 1)
    
    actual_n_bins = len(time_bins) - 1
    total_events = np.zeros(actual_n_bins)
    total_opps = np.zeros(actual_n_bins)
    
    for times in all_trees_times:
        events, opps = calc_skyline_components(times, time_bins)
        total_events += events
        total_opps += opps
        
    N_t = np.divide(total_opps, total_events, out=np.full_like(total_opps, np.nan), where=total_events > 0)
    
    # --- SFS Probabilities Calculation ---
    ks = sorted(list(set(total_obs_sfs.keys()) | set(total_exp_sfs.keys())))
    obs_total = sum(total_obs_sfs.values()) if total_obs_sfs else 1
    exp_total = sum(total_exp_sfs.values()) if total_exp_sfs else 1
    
    obs_probs = [total_obs_sfs.get(k, 0) / obs_total for k in ks]
    exp_probs = [total_exp_sfs.get(k, 0) / exp_total for k in ks]

    # --- Plotting Panel ---
    print(f"Plotting results to {args.output}...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Demographic History
    ax1 = axes[0]
    ax1.stairs(N_t, time_bins, fill=True, alpha=0.3, color='tab:blue')
    ax1.stairs(N_t, time_bins, baseline=None, color='tab:blue', linewidth=2)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("time (generations)")
    ax1.set_ylabel("$N_e(t)$")
    ax1.set_title("Inferred $N_e(t)$ for D. melanogaster")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # Panel 2: Folded SFS Comparison
    ax2 = axes[1]
    if sum(total_obs_sfs.values()) > 0:
        ax2.plot(ks, obs_probs, marker='o', linestyle='', color='black', label='observed', alpha=0.7)
    if sum(total_exp_sfs.values()) > 0:
        ax2.plot(ks, exp_probs, marker='s', linestyle='-', color='tab:orange', label='expected (trees)', alpha=0.8)
        
    ax2.set_yscale('log')
    ax2.set_xlabel("minor allele count ($k$)")
    ax2.set_ylabel("log prob")
    ax2.set_title("SFS comparison")
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print("Done!")

if __name__ == "__main__":
    main()