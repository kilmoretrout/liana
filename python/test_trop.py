import os
import torch
import numpy as np
import msprime
import tskit
from torch.utils.cpp_extension import load

def compile_module():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_src = os.path.join(cur_dir, "tropical_kernel.cu")
    
    # Optional: Clear the JIT cache to ensure a fresh build
    # import shutil
    # shutil.rmtree(os.path.expanduser("~/.cache/torch_extensions/py310_cu132"), ignore_errors=True)
    
    print("JIT compiling unified CUDA kernel...")
    return load(
        name="tropical_gcn",
        sources=[cuda_src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False
    )

def extract_tskit_tensors(ts):
    """
    Converts a tskit TreeSequence into batched, padded tensors 
    ordered topologically (post-order) for the CUDA kernel.
    """
    num_trees = ts.num_trees
    num_nodes = ts.num_nodes
    
    # Store edges and branch lengths per tree
    tree_edges = []
    tree_bls = []
    
    max_edges = 0
    
    for tree in ts.trees():
        edges = []
        bls = []
        
        # We MUST extract edges in post-order to satisfy the CUDA thread sequential loop
        for u in tree.nodes(order="postorder"):
            p = tree.parent(u)
            if p != tskit.NULL:
                edges.append([u, p])
                bls.append(tree.time(p) - tree.time(u))
                
        tree_edges.append(edges)
        tree_bls.append(bls)
        max_edges = max(max_edges, len(edges))
        
    # Create padded numpy arrays
    # -1 is used as the pad token, which the CUDA kernel ignores
    edges_np = np.full((num_trees, max_edges, 2), -1, dtype=np.int64)
    bl_np = np.zeros((num_trees, max_edges), dtype=np.float32)
    
    for b in range(num_trees):
        E = len(tree_edges[b])
        if E > 0:
            edges_np[b, :E, :] = tree_edges[b]
            bl_np[b, :E] = tree_bls[b]
            
    return edges_np, bl_np, num_nodes

def cpu_reference(edges, branch_lengths, leaf_states, num_nodes):
    B, E, _ = edges.shape
    _, L, C = leaf_states.shape
    
    node_states = np.full((B, num_nodes, C), -np.inf, dtype=np.float32)
    node_states[:, :L, :] = leaf_states
    
    for b in range(B):
        for e in range(E):
            c = edges[b, e, 0]
            p = edges[b, e, 1]
            if c < 0 or p < 0:
                continue
            bl = branch_lengths[b, e]
            cand = node_states[b, c] + bl
            node_states[b, p] = np.maximum(node_states[b, p], cand)
            
    return node_states

def main():
    if not torch.cuda.is_available():
        raise SystemError("CUDA GPU required.")

    module = compile_module()

    # 1. Simulate data using msprime
    # We use 32 samples (16 diploid equivalents), with recombination to generate a tree sequence
    print("Simulating tree sequence with msprime...")
    ts = msprime.sim_ancestry(
        samples=16, 
        sequence_length=1e6, # 1 Mb window
        recombination_rate=1e-8,
        population_size=1e4,
        random_seed=42
    )
    
    num_trees = ts.num_trees
    num_leaves = ts.num_samples
    print(f"Generated {num_trees} marginal trees across the 1Mb sequence.")
    
    # 2. Extract topology into padded tensors
    print("Extracting topologies and padding tensors...")
    edges_np, bl_np, total_nodes = extract_tskit_tensors(ts)
    
    edges_t = torch.tensor(edges_np, device="cuda")
    bl_t = torch.tensor(bl_np, device="cuda")
    
    # 3. Generate mock leaf states (e.g., site pattern embeddings or VCF counts)
    channels = 64
    torch.manual_seed(42)
    # Shape: [batch_trees, num_samples, channels]
    leaf_states_t = torch.randn(num_trees, num_leaves, channels, dtype=torch.float32, device="cuda")
    
    # 4. Execute CUDA Kernel
    print(f"Launching CUDA kernel for {num_trees} trees on {channels} channels...")
    out_cuda = module.forward(edges_t, bl_t, leaf_states_t, total_nodes)
    out_cuda = torch.where(torch.isneginf(out_cuda), torch.zeros_like(out_cuda), out_cuda)
    
    torch.cuda.synchronize()
    
    # 5. Verify against CPU
    print("Verifying against CPU reference...")
    ref_out = cpu_reference(edges_np, bl_np, leaf_states_t.cpu().numpy(), total_nodes)
    
    ref_out[np.isneginf(ref_out)] = 0.0
    
    out_cuda_np = out_cuda.cpu().numpy()
    print(out_cuda_np, ref_out)
    max_err = np.max(np.abs(out_cuda_np - ref_out))
    
    print(f"Max absolute discrepancy: {max_err:.6e}")
    assert np.allclose(out_cuda_np, ref_out, atol=1e-5), "CUDA computation diverged from CPU."
    print("Test SUCCESS: The CUDA kernel correctly executed the tropical max-plus operations natively on the tskit TreeSequence.")

if __name__ == "__main__":
    main()