# -*- coding: utf-8 -*-
import argparse
import os
import glob
import gzip
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def count_segregating_sites(vcf_path, target_ind, window_size):
    """
    Parses a VCF and counts segregating sites per window.
    Includes zero-filling for windows with no variation.
    Masked/missing values are not considered unique alleles.
    """
    open_func = gzip.open if vcf_path.endswith('.gz') else open
    
    window_counts = {}
    max_pos = 0.0
    
    
    
    with open_func(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'): continue
        
            #print(line)
            
            cols = line.strip().split('\t')
            if len(cols) < 9 + target_ind: continue
            
            pos = float(cols[1])
            if pos > max_pos: 
                max_pos = pos
                
            gts = cols[9:9+target_ind]
            
            first_valid_val = None
            is_segregating = False
            
            for gt_str in gts:
                gt_base = gt_str.split(':')[0]
                
                # Skip masked/missing genotypes entirely
                if '.' in gt_base:
                    continue
                
                # Map to standard values: 1.0 if '1' is present, else 0.0
                val = 1.0 if '1' in gt_base else 0.0
                
                if first_valid_val is None:
                    first_valid_val = val
                elif val != first_valid_val:
                    is_segregating = True
                    break # EARLY EXIT: Found valid variation, move to next site
                    
            if is_segregating:
                win_idx = int(pos // window_size)
                window_counts[win_idx] = window_counts.get(win_idx, 0) + 1
                
    # Zero-fill: Ensure we include windows that had 0 segregating sites
    max_win_idx = int(max_pos // window_size)
    full_counts = [window_counts.get(w, 0) for w in range(max_win_idx + 1)]
    
    return full_counts

def main():
    parser = argparse.ArgumentParser(description="Plot Histogram of Segregating Sites per Window")
    parser.add_argument("--vcf_dir", required=True, help="Directory containing .vcf or .vcf.gz files")
    parser.add_argument("--target_ind", type=int, default=50, help="Number of haploid individuals to analyze")
    parser.add_argument("--window_size", type=int, default=2_000_000, help="Window size in base pairs (default: 1Mb)")
    parser.add_argument("--output", default="segregating_sites_hist.png", help="Output plot filename")
    args = parser.parse_args()

    vcf_files = glob.glob(os.path.join(args.vcf_dir, "*.vcf.gz")) + glob.glob(os.path.join(args.vcf_dir, "*.vcf"))
    
    if not vcf_files:
        raise ValueError(f"No VCF files found in {args.vcf_dir}")
        
    print(f"Found {len(vcf_files)} VCF files. Processing...")
    
    all_window_counts = []
    
    for vcf in tqdm(vcf_files, desc="Parsing VCFs"):
        counts = count_segregating_sites(vcf, args.target_ind, args.window_size)
        all_window_counts.extend(counts)
        
    all_window_counts = np.array(all_window_counts)
    
    # Calculate summary statistics
    mean_sites = np.mean(all_window_counts)
    median_sites = np.median(all_window_counts)
    total_windows = len(all_window_counts)
    
    print(f"\nAnalyzed {total_windows} total windows.")
    print(f"Mean segregating sites/window: {mean_sites:.2f}")
    print(f"Median segregating sites/window: {median_sites:.2f}")

    print(np.min(all_window_counts))
    # Format window size string for plots (e.g., "1 Mb" or "500 kb")
    if args.window_size >= 1_000_000:
        win_str = f"{args.window_size / 1_000_000:g} Mb"
    else:
        win_str = f"{args.window_size / 1_000:g} kb"

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    print(f"Plotting histogram to {args.output}...")
    plt.figure(figsize=(10, 6))
    
    # Use auto-binning, but ensure bins align nicely with integer counts
    plt.hist(all_window_counts, bins='auto', color='tab:blue', alpha=0.7, edgecolor='black')
    
    # Add vertical lines for mean and median
    plt.axvline(mean_sites, color='tab:red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_sites:.0f}')
    plt.axvline(median_sites, color='tab:green', linestyle='dotted', linewidth=2, label=f'Median: {median_sites:.0f}')
    
    plt.xlabel(f"Number of Segregating Sites per {win_str} Window")
    plt.ylabel("Frequency (Number of Windows)")
    plt.title(f"Segregating Sites Distribution ({args.target_ind} Haploids)")
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print("Done!")

if __name__ == "__main__":
    main()
