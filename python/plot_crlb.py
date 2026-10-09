import numpy as np
import matplotlib.pyplot as plt

def harmonic_number(n):
    return sum(1.0 / i for i in range(1, n))

def analytical_epoch_bound(n_leaves, mu, rho):
    H = harmonic_number(n_leaves)
    
    # The spatial info depends entirely on the ratio
    p_survive = np.exp(-rho / mu)
    I_spatial = p_survive / (1.0 - p_survive)
    
    k_values = np.arange(n_leaves, 1, -1)
    variances = np.zeros(len(k_values))
    
    for idx, k in enumerate(k_values):
        # 1. Expected Data Information via Watterson
        I_data = 1.0 / ((k - 1) * H)
        
        # 2. Asymptotic Toeplitz Inverse
        var_tau = 1.0 / np.sqrt(I_data**2 + 4 * I_data * I_spatial)
        variances[idx] = var_tau
        
    return k_values, np.sqrt(variances)

# Compare different recombination regimes
n = 15
k_vals, std_human = analytical_epoch_bound(n, mu=1.5e-8, rho=1.0e-8)  # ρ < μ (Human avg)
_, std_hotspot = analytical_epoch_bound(n, mu=1.5e-8, rho=1.5e-7)     # ρ > μ (Hotspot)

print(f"Analytical CRLB Standard Error (log-units) for n={n}")
print("Epoch(k) | Human Avg (ρ<μ) | Hotspot (ρ>μ)")
print("-" * 45)
for i in range(len(k_vals)):
    print(f"  k={k_vals[i]:<4} |    ±{std_human[i]:.4f}     |   ±{std_hotspot[i]:.4f}")