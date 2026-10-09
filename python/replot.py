# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Replot N(t) and SFS from a saved .npz file")
    parser.add_argument("--input", default="results.npz", help="Path to the saved .npz file")
    parser.add_argument("--output", default="adjusted_plot.png", help="Output plot filename")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    try:
        data = np.load(args.input)
    except FileNotFoundError:
        print(f"Error: Could not find {args.input}. Please ensure the file exists.")
        return

    # Extract all arrays
    time_bins = data['time_bins']
    N_t_base = data['N_t_base']
    lower_ci = data['lower_ci']
    upper_ci = data['upper_ci']
    k_vals = data['k_vals']
    obs_probs = data['obs_probs']
    exp_probs = data['exp_probs']
    log_likelihood, kl_divergence = data['metrics']

    print(f"Plotting results to {args.output}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ---------------------------------------------------------
    # Panel 1: N(t) Skyline
    # ---------------------------------------------------------
    # Append the last value to the CI arrays so fill_between(step='post') covers the final bin
    ax1.fill_between(
        time_bins, 
        np.append(lower_ci, lower_ci[-1]), 
        np.append(upper_ci, upper_ci[-1]),
        step='post', alpha=0.3, color='tab:blue', label="95% CI"
    )
    ax1.stairs(N_t_base, time_bins, baseline=None, color='tab:blue', linewidth=2, label="MLE N(t)")
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    #ax1.set_ylim(1e5, 1e8) # Adjust this if your N_e range is different
    
    ax1.set_xlabel("time (generations)")
    ax1.set_ylabel("$N_e(t)$")
    ax1.set_title("Demographic History from GCN-UNet Trees")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()
    
    # ---------------------------------------------------------
    # Panel 2: SFS Line Plot (Log Space)
    # ---------------------------------------------------------
    ax2.set_yscale('log')
    
    ax2.plot(k_vals, obs_probs, marker='o', linestyle='-', linewidth=2, 
             label='Observed (VCF)', color='tab:gray', markersize=6)
    ax2.plot(k_vals, exp_probs, marker='s', linestyle='--', linewidth=2, 
             label='Implied (Trees)', color='tab:orange', markersize=6)
    
    ax2.set_xlabel("Minor Allele Count (k)")
    ax2.set_ylabel("Log Probability (Proportion)")
    
    title_str = "Folded SFS: Observed vs Implied\n"
    title_str += f"Log-Likelihood: {log_likelihood:,.0f} | KL-Divergence: {kl_divergence:.4f}"
    ax2.set_title(title_str)
    
    # Force integer ticks on the X-axis for allele counts
    ax2.set_xticks(k_vals)
    
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()
    
    # Finalize and save
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print("Done!")

if __name__ == "__main__":
    main()
