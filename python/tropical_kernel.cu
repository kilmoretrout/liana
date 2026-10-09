#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// 1. The CUDA Kernel
template <typename scalar_t>
__global__ void tropical_forward_kernel(
    const int64_t* __restrict__ edges,           
    const scalar_t* __restrict__ branch_lengths, 
    scalar_t* __restrict__ node_states,          
    int64_t batch_size,
    int64_t num_edges,
    int64_t num_nodes,
    int64_t channels) 
{
    int64_t b = blockIdx.x; 
    int64_t c = threadIdx.x; 
    
    if (b < batch_size && c < channels) {
        for (int64_t e = 0; e < num_edges; ++e) {
            int64_t edge_idx = b * num_edges * 2 + e * 2;
            int64_t child  = edges[edge_idx];
            int64_t parent = edges[edge_idx + 1];
            
            if (child < 0 || parent < 0) continue; 
            
            scalar_t bl = branch_lengths[b * num_edges + e];
            scalar_t child_val = node_states[(b * num_nodes + child) * channels + c];
            
            scalar_t candidate = child_val + bl;
            
            int64_t parent_offset = (b * num_nodes + parent) * channels + c;
            scalar_t cur = node_states[parent_offset];
            
            if (candidate > cur) {
                node_states[parent_offset] = candidate;
            }
        }
    }
}

// 2. The PyTorch C++ Wrapper
torch::Tensor tropical_forward_cuda(
    torch::Tensor edges, 
    torch::Tensor branch_lengths, 
    torch::Tensor leaf_states, 
    int64_t num_nodes) 
{
    TORCH_CHECK(edges.is_cuda(), "edges must be on CUDA");
    TORCH_CHECK(branch_lengths.is_cuda(), "branch_lengths must be on CUDA");
    TORCH_CHECK(leaf_states.is_cuda(), "leaf_states must be on CUDA");
    
    auto edges_c = edges.contiguous();
    auto bl_c = branch_lengths.contiguous();
    auto ls_c = leaf_states.contiguous();
    
    int64_t batch_size = edges_c.size(0);
    int64_t num_edges  = edges_c.size(1);
    int64_t num_leaves = ls_c.size(1);
    int64_t channels   = ls_c.size(2);
    
    auto options = ls_c.options();
    torch::Tensor node_states = torch::full(
        {batch_size, num_nodes, channels},
        -std::numeric_limits<float>::infinity(),
        options
    );
    
    node_states.slice(1, 0, num_leaves).copy_(ls_c);

    int threads = channels; 
    int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(ls_c.scalar_type(), "tropical_forward_cuda", ([&] {
        tropical_forward_kernel<scalar_t><<<blocks, threads>>>(
            edges_c.data_ptr<int64_t>(),
            bl_c.data_ptr<scalar_t>(),
            node_states.data_ptr<scalar_t>(),
            batch_size,
            num_edges,
            num_nodes,
            channels
        );
    }));
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return node_states;
}

// 3. The PyBind11 Module Definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tropical_forward_cuda, "Tropical GCN Forward Pass (CUDA)");
}