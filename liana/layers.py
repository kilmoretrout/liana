# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dist_functions import PdistLorentz, tangent_to_lorentz

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Standard sine/cosine positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class ConditionalARGModel(nn.Module):
    def __init__(self, vocab_size, latent_dim=128, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. Latent Projection: Maps continuous z vector to Transformer's embedding dimension
        self.latent_projection = nn.Linear(latent_dim, d_model)
        
        # 2. Token Embedding & Positional Encoding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer backbone (Using Encoder with causal masking to act as Decoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output projection to vocabulary space
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, src, z):
        """
        src: (batch_size, seq_len) - The discrete ARG token sequence
        z:   (batch_size, latent_dim) - The continuous latent condition vector
        """
        batch_size, seq_len = src.size()
        device = src.device
        
        # -- A. Process Latent Vector --
        # Project z and unsqueeze to treat it as the first token in the sequence
        # Shape: (batch_size, 1, d_model)
        z_proj = self.latent_projection(z).unsqueeze(1) 
        
        # -- B. Process Discrete Sequence --
        # Embed discrete ARG tokens
        # Shape: (batch_size, seq_len, d_model)
        x_emb = self.embedding(src) * math.sqrt(self.d_model)
        
        # -- C. Combine & Apply Positional Encoding --
        # Concatenate the latent token and the token embeddings
        # New Shape: (batch_size, seq_len + 1, d_model)
        x = torch.cat([z_proj, x_emb], dim=1)
        
        # Apply positional encoding across the combined sequence
        x = self.pos_encoder(x)
        
        # -- D. Create Masks --
        # Causal mask ensures tokens can only attend to the latent vector and past tokens
        # Size: (seq_len + 1, seq_len + 1)
        causal_mask = self.generate_square_subsequent_mask(seq_len + 1).to(device)
        
        # Padding mask to ignore <PAD> tokens in the discrete sequence (index 0)
        # We prepend a 'False' for the latent token, because the latent token is never padded
        # Shape: (batch_size, seq_len + 1)
        src_padding_mask = (src == 0)
        latent_padding_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([latent_padding_mask, src_padding_mask], dim=1)
        
        # -- E. Transformer Pass --
        output = self.transformer(
            x, 
            mask=causal_mask, 
            src_key_padding_mask=key_padding_mask,
            is_causal=True
        )
        
        # -- F. Output Generation --
        # Discard the first position of the output (corresponding to the z token)
        # because we only want to predict the next discrete ARG tokens based on the sequence.
        # Output shape after slicing: (batch_size, seq_len, d_model)
        output_seq = output[:, 1:, :] 
        
        # Map to vocabulary size for cross-entropy
        # Final shape: (batch_size, seq_len, vocab_size)
        logits = self.fc_out(output_seq)
        
        return logits

class HaarSWT(nn.Module):
    def __init__(self, levels=8):
        """
        levels (int): The number of scales 'c' you want in your tensor.
                      A level of 8 gives scales equivalent to 2, 4, 8... 256.
        """
        super().__init__()
        self.levels = levels
        
        # Haar low-pass (scaling) and high-pass (wavelet) filters
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        
        # Shape required for PyTorch conv1d: (out_channels, in_channels, kernel_size)
        self.register_buffer('h0', torch.tensor([[[inv_sqrt2, inv_sqrt2]]]))
        self.register_buffer('h1', torch.tensor([[[inv_sqrt2, -inv_sqrt2]]]))

    def forward(self, x):
        """
        Args:
            x: Binary mutation matrix of shape (n, l) or (batch, n, l)
        Returns:
            feature_tensor: Shape (levels + 1, n, l)
        """
        # Ensure x is 3D for conv1d: (batch_size, channels, length)
        # If input is (n, l), we treat it as (n, 1, l) where each sequence is processed independently
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        elif x.dim() == 3:
            # If (batch, n, l), fold batch and n together temporarily
            b, n, l = x.shape
            x = x.view(b * n, 1, l)
            
        approx = x
        details = []
        
        for i in range(self.levels):
            # The 'à trous' magic: exponentially dilate the filter instead of decimating the signal
            dilation = 2 ** i
            
            # To keep the output length exactly 'l', we must pad the input.
            # Effective kernel size is: 1 + (kernel_size - 1) * dilation
            # For Haar (kernel_size=2), effective size is 1 + dilation.
            # We use 'replicate' padding on the left to avoid artificial zero-boundary artifacts.
            approx_padded = F.pad(approx, (dilation, 0), mode='replicate')
            
            # Apply the dilated filters
            detail = F.conv1d(approx_padded, self.h1, dilation=dilation)
            approx = F.conv1d(approx_padded, self.h0, dilation=dilation)
            
            details.append(detail)
            
        # details is a list of 'c' tensors of shape (N, 1, l)
        # approx is the final low-pass remainder of shape (N, 1, l)
        
        # 1. Concatenate the final approximation (Channel 0) with all detail scales
        # This explicitly gives the U-Net the total branch length (mean density) + the structural breaks
        all_channels = [approx] + details[::-1] # Reverse details so smallest scale is index 1
        
        # 2. Stack along a new channel dimension
        # Shape becomes (N, c+1, l)
        feature_tensor = torch.cat(all_channels, dim=1)
        
        # Unfold the batch dimension if it was provided
        if 'b' in locals():
            feature_tensor = feature_tensor.view(b, n, self.levels + 1, -1)
            # Permute to standard (batch, c, n, l) format
            feature_tensor = feature_tensor.permute(0, 2, 1, 3) 
        else:
            # Permute to (c+1, n, l)
            feature_tensor = feature_tensor.permute(1, 0, 2)
            
        return feature_tensor
    
# gcn stuff
# ================
    
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.typing import Adj, OptTensor, PairTensor
from typing import Union, Tuple, Optional

from hypll.manifolds.poincare_ball import PoincareBall, Curvature
import hypll.nn as hnn

from hypll.tensors import ManifoldTensor, TangentTensor

# Assuming MessageNorm is defined or imported as provided in your snippet:
class MessageNorm(torch.nn.Module):
    def __init__(self, learn_scale: bool = False, device: Optional[torch.device] = None):
        super().__init__()
        self.scale = Parameter(torch.empty(1, device=device), requires_grad=learn_scale)
        self.reset_parameters()

    def reset_parameters(self):
        self.scale.data.fill_(1.0)

    def forward(self, x: Tensor, msg: Tensor, p: float = 2.0) -> Tensor:
        msg = F.normalize(msg, p=p, dim=-1)
        x_norm = x.norm(p=p, dim=-1, keepdim=True)
        return msg * x_norm * self.scale

class GATv2Conv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        # ### NEW: Add arguments to toggle MessageNorm
        add_msg_norm: bool = False, 
        learn_msg_scale: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights
        
        # ### NEW: Initialize MessageNorm
        self.add_msg_norm = add_msg_norm
        if add_msg_norm:
            self.msg_norm = MessageNorm(learn_scale=learn_msg_scale)
        else:
            self.msg_norm = None

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        # ### NEW: Reset MessageNorm
        if self.msg_norm is not None:
            self.msg_norm.reset_parameters()
        
        # Standard GAT initialization
        import torch.nn.init as init
        init.xavier_uniform_(self.att)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ):
        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        x_l: OptTensor = None
        x_r: OptTensor = None
        
        # Identify input for normalization (target nodes)
        x_in = x 

        if isinstance(x, Tensor):
            assert x.dim() == 2
            if self.res is not None:
                res = self.res(x)
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            x_in = x[1] # For bipartite, target is at index 1
            assert x[0].dim() == 2
            if x_r is not None and self.res is not None:
                res = self.res(x_r)
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
             if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)

        alpha = self.edge_updater(edge_index, x=(x_l, x_r),
                                  edge_attr=edge_attr)

        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        # 1. Collapse heads first
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # ### NEW: Apply MessageNorm before adding residual
        if self.msg_norm is not None:
            # We use x_in (the original node features) to scale the aggregated message
            out = self.msg_norm(x_in, out)

        # 2. Add Residual
        if res is not None:
            out = out + res

        # 3. Add Bias
        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
             # (Keeping original logic for return_attention_weights...)
             if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
             else: # SparseTensor
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        # Standard GATv2 edge_update
        x = x_i + x_j
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            if self.lin_edge is not None:
                edge_attr = self.lin_edge(edge_attr)
                edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
                x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)

import torch_geometric

class ResConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_w = 3, dilations = [1, 1, 1]):
        super().__init__()
        
        self.conv0 = nn.Conv2d(in_dim, out_dim, kernel_size = (1, kernel_w), padding = (0, ((kernel_w - 1) * (dilations[0]) // 2)), stride = (1, 1), dilation = dilations[0])
        self.norm0 = nn.InstanceNorm2d(out_dim)
        
        self.conv1 = nn.Conv2d(out_dim, out_dim, kernel_size = (1, kernel_w), padding = (0, ((kernel_w - 1) * (dilations[1]) // 2)), stride = (1, 1), dilation = dilations[1])
        self.norm1 = nn.InstanceNorm2d(out_dim)
        
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size = (1, kernel_w), padding = (0, ((kernel_w - 1) * (dilations[2]) // 2)), stride = (1, 1), dilation = dilations[2])
    
        self.act = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.act(self.norm0(self.conv0(x)))
        
        x = self.conv2(x + self.act(self.norm1(self.conv1(x))))
        
        return x
    
class GCNBlock(nn.Module):
    def __init__(self, in_dim=104, n_gcn_layers=2, n_heads=4, K=5, dropout=0.0, norm=nn.LayerNorm, 
                 add_msg_norm=False, learn_msg_scale=True):
        super().__init__()
        
        self.K = K
        
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = nn.LeakyReLU()
        
        for ix in range(n_gcn_layers):    
            self.norms.append(norm((in_dim,)))
            self.gcns.append(GATv2Conv(in_dim, in_dim // n_heads, heads=n_heads, dropout=dropout, 
                                       share_weights=True, add_msg_norm=add_msg_norm, learn_msg_scale=learn_msg_scale))

        self.linear = nn.Linear(in_dim, in_dim)
    
    def forward(self, x, edge_index=None, random_edges=False):
        n = x.shape[1]
        
        if edge_index is None:
            # x: (B, N, D)
            B, N, D = x.shape
            
            # --- CRITICAL FIX ---
            # You cannot have more neighbors than nodes in the graph.
            # This protects the GCN when GraphPool shrinks N < K.
            actual_k = min(self.K, N)
            
            x_flat = x.reshape(B * N, D)
            
            if random_edges:
                # 1. Target nodes: repeat 'actual_k' times instead of self.K
                targets = torch.arange(B * N, device=x.device).repeat_interleave(actual_k)
                
                # 2. Batch offsets
                batch_offsets = (targets // N) * N
                
                # 3. Generate random local source nodes 
                local_sources = torch.randint(0, N, (B * N * actual_k,), device=x.device)
                
                # 4. Avoid self-loops
                local_targets = targets % N
                local_sources = torch.where(
                    local_sources == local_targets, 
                    (local_sources + 1) % N, 
                    local_sources
                )
                
                # 5. Apply batch offset 
                sources = batch_offsets + local_sources
                
                # PyG format
                edge_index = torch.stack([sources, targets], dim=0)
                
            else:
                # Create batch vector
                batch = torch.arange(B, device=x.device).repeat_interleave(N)
                
                # Compute KNN graph for entire batch at once
                dists = torch.cdist(x, x)
        
                # 2. Mask self-loops with infinity
                dists.diagonal(dim1=1, dim2=2).fill_(float('inf'))
                
                # 3. Get local indices of actual_k nearest neighbors
                _, local_sources = torch.topk(dists, k=actual_k, dim=-1, largest=False)
                
                # 4. Add batch offsets
                batch_offsets = (torch.arange(B, device=x.device) * N).view(B, 1, 1)
                global_sources = (local_sources + batch_offsets).flatten()
                
                # 5. Create target indices expanding to actual_k
                global_targets = torch.arange(B * N, device=x.device).view(-1, 1).expand(-1, actual_k).flatten()
                
                # 6. Stack into PyG format - BOTH arrays are now guaranteed the same size
                edge_index = torch.stack([global_sources, global_targets], dim=0)
                
        x = x.reshape(-1, x.shape[-1])
        in_dim = x.shape[-1]
                
        # forward
        for ix in range(len(self.gcns) - 1):
            x = self.norms[ix](self.gcns[ix](x, edge_index) + x)    
            x = self.act(x)

        x = self.norms[-1](self.gcns[-1](x, edge_index) + x)   

        x = x.reshape(-1, n, in_dim)
        
        return x, edge_index
    
class SingleConv(nn.Module):
    def __init__(self, in_channels = 130, out_channels = 32, kernel_w = 3):
        super().__init__()
        self.conv = ResConvBlock(in_channels, out_channels, kernel_w = kernel_w)
        
    def forward(self, x):
        return self.conv(x)
    
class ConvGCNBlock(nn.Module):
    def __init__(self, in_dim=130, out_dim=32, K=3, eps=0.):
        super().__init__()
        
        self.eps = eps
        
        self.conv = SingleConv(in_dim, out_dim)
        self.gcn = GCNBlock(out_dim, K=K)
        self.act = nn.LeakyReLU()
    
    def forward(self, x, edge_index=None):
        x = self.conv(x)
                
        # Capture the dynamic Batch dimension 'B'
        B, c, n, s = x.shape
        
        # Transpose to (B, s, n, c)
        x = x.transpose(3, 1)
        
        # Flatten B and s into a single dimension to feed the GCN block: (B*s, n, c)
        x = x.reshape(-1, n, c) 
        
        x, edge_index = self.gcn(x, edge_index)
        
        # Reshape back to (B, s, n, c) using the captured batch size
        x = x.reshape(B, s, n, c)
        
        # Transpose back to original structure (B, c, n, s)
        x = x.transpose(3, 1)
        
        return x
    
from dist_functions import Pdist, PdistL2, PdistPoincare

def to_tangent(x: ManifoldTensor, manifold) -> torch.Tensor:
    """
    Safely maps a ManifoldTensor to the tangent space at the origin.
    Returns a raw PyTorch tensor.
    """
    # hypll strictly requires 'y' to be the ManifoldTensor object itself, not the raw tensor!
    tangent_obj = manifold.logmap(x=None, y=x)
    return tangent_obj.tensor

def to_manifold(v_tensor: torch.Tensor, manifold, man_dim: int = -1) -> ManifoldTensor:
    """
    Safely maps a raw PyTorch tensor to the manifold.
    - Use man_dim = -1 for GCNs, Linear layers, and 1D features (Default)
    - Use man_dim = 1 for 2D Convolutions (B, C, H, W)
    """
    tangent_obj = TangentTensor(data=v_tensor, man_dim=man_dim, manifold=manifold)
    return manifold.expmap(tangent_obj)

def add_pop_features(x):
    """
    Adds a single binary population indicator channel to a 4D tensor.

    This function takes a tensor of shape (b, c, n, l) and appends 1 extra
    channel. In this new channel, the first n // 2 elements along the 'n' 
    dimension are set to 0, and the remaining elements are set to 1.

    Args:
        x (torch.Tensor): The input tensor of shape (b, c, n, l).

    Returns:
        torch.Tensor: The output tensor of shape (b, c + 1, n, l)
                      with the added binary feature channel.
    """
    b, c, n, l = x.shape
    
    # The splitting logic requires the 'n' dimension to be an even number.
    if n % 2 != 0:
        raise ValueError("The 'n' dimension must be an even number to split evenly.")

    # --- 1. Create the Binary Channel ---
    # Create a tensor of zeros with shape (1, 1, n, 1) matching the device and dtype of x
    pop_channel = torch.zeros((1, 1, n, 1), device=x.device, dtype=x.dtype)
    
    # Set the second half of the 'n' dimension to 1
    pop_channel[:, :, n // 2:, :] = 1.0
    
    # --- 2. Expand the Channel Tensor ---
    # Broadcast the tensor to match the full 4D shape of x for concatenation.
    # The `-1` tells expand to keep the original size for that dimension.
    # New shape will be (b, 1, n, l).
    pop_channel_expanded = pop_channel.expand(b, -1, -1, l)
    
    # --- 3. Concatenate with the Original Tensor ---
    # Concatenate the original tensor 'x' and the new binary channel
    # along the channel dimension (dim=1).
    result = torch.cat([x, pop_channel_expanded], dim=1)
    
    return result

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Standard sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Add positional encodings sliced to the current sequence length
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x

class BiLSTM(nn.Module):
    def __init__(self, c_features, hidden_dim, num_layers=2):
        """
        Args:
            c_features: The number of input and output features per site (c).
            hidden_dim: The internal hidden state size for the LSTM.
            num_layers: Number of stacked Bi-LSTM layers.
        """
        super().__init__()
        
        # Bidirectional LSTM: outputs 2 * hidden_dim features per sequence step
        self.bilstm = nn.LSTM(
            input_size=c_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Linear layer to project the concatenated forward/backward 
        # hidden states back down to the original 'c' dimension
        self.projection = nn.Linear(hidden_dim * 2, c_features)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (c, n, l)
               c = features, n = samples, l = sequence length (sites)
        Returns:
            out: Output tensor of shape (c, n, l)
        """
        x = x.permute(1, 0, 2)
                
        # STEP 2: Pass through the Bi-LSTM
        # lstm_out shape: (n, l, 2 * hidden_dim)
        # We ignore the hidden/cell state tuples (h_n, c_n) for this pass
        lstm_out, _ = self.bilstm(x)
        
        # STEP 3: Project back to the original feature dimension 'c'
        # linear_out shape: (n, l, c)
        final_out = self.projection(lstm_out)
        
        return final_out.permute(1, 0, 2)
        
    
import torch
import torch.nn as nn

class GraphPool(nn.Module):
    """ Learns to pool nodes by keeping the top N//2 most activated nodes. """
    def __init__(self, in_channels):
        super().__init__()
        # Increased capacity for scoring
        self.score_network = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(in_channels // 2, 1)
        )
        
        # If you have an edge_index, you could replace the above with a GNN layer:
        # self.score_network = GATv2Conv(in_channels, 1)

    def forward(self, x, edge_index=None):
        # x shape: (B, C, N, L)
        B, C, N, L = x.shape
        k = max(1, N // 2)
        
        # 1. Compute node scores (Averaging over L preserves topology across sequence)
        x_mean = x.mean(dim=-1).transpose(1, 2) # (B, N, C)
        
        # [Optional] If using a GNN for scoring: scores = self.score_network(x_mean, edge_index)
        scores = self.score_network(x_mean).squeeze(-1) # (B, N)
        
        # 2. Get top N // 2 nodes
        topk_scores, topk_indices = torch.topk(scores, k, dim=-1) # (B, k)
        
        # 3. Gather features of the kept nodes
        idx_expanded = topk_indices.unsqueeze(1).unsqueeze(-1).expand(B, C, k, L)
        x_pooled = torch.gather(x, 2, idx_expanded)
        
        # 4. Gate features to route gradients, using tanh for better variance bounds
        # Adding an epsilon or scaling by (scores / norm) prevents severe magnitude decay
        scores_expanded = torch.tanh(topk_scores).unsqueeze(1).unsqueeze(-1)
        
        # Divide by a constant or norm to stabilize variance, or simply use tanh
        x_pooled = x_pooled * scores_expanded 
        
        # 5. [Crucial] Coarsen the graph topology
        # pooled_edge_index = filter_adjacency(edge_index, topk_indices)
        
        return x_pooled, topk_indices, N # Add pooled_edge_index to returns


class GraphUnpool(nn.Module):
    """ Scatters pooled nodes back to their original indices. """
    def __init__(self):
        super().__init__()

    def forward(self, x, indices, orig_N):
        # x shape: (B, C, k, L)
        # indices shape: (B, k)
        B, C, k, L = x.shape
        
        # Create zero tensor for the original graph size
        out = torch.zeros((B, C, orig_N, L), device=x.device, dtype=x.dtype)
        
        # Scatter features precisely back to their original locations
        idx_expanded = indices.unsqueeze(1).unsqueeze(-1).expand(B, C, k, L)
        out.scatter_(2, idx_expanded, x)
        
        return out


class ConvGCNUNet(nn.Module):
    def __init__(self, in_dim=130, init_dim=12,
                 coal_dim=15, K=5, embedding_dim=14,
                 gcn_dims=[512, 256, 128, 64], gru_dim=512, two_pop=False, 
                 eps=1e-8, dist="poincare", return_rec_dist=False):
        super().__init__()
        
        # --- General Parameters ---
        self.two_pop = two_pop
        if self.two_pop:
            in_dim += 1
        
        self.dist = dist
        if len(gcn_dims) != 4:
            raise ValueError("gcn_dims must contain 4 channel sizes for the 4 U-Net levels.")

        # --- New regime switch ---
        self.return_rec_dist = return_rec_dist

        # --- Initial Convolution ---
        #self.init_conv = nn.Conv2d(in_dim, gcn_dims[0], (1, 1), stride=(1, 1), bias=False)

        # --- Encoder (Downsampling Path) ---
        self.encoders = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        
        self.alpha = nn.Parameter(torch.ones(1), requires_grad = True)
        self.beta = nn.Parameter(torch.zeros(1), requires_grad = True)
        
        in_channels = in_dim
        for num_channels in gcn_dims:
            # Add the main block for this level
            self.encoders.append(ConvGCNBlock(in_channels, num_channels, K=K))
            in_channels = num_channels
            
            # Add a downsampling block for all but the last level
            if num_channels != gcn_dims[-1]:
                self.down_convs.append(
                    nn.Conv2d(num_channels, num_channels, kernel_size=(1,2), stride=(1,2)
                ))

        # --- Decoder (Upsampling Path) ---
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Iterate through encoder channels in reverse order
        reversed_gcn_dims = gcn_dims[::-1]
        for i in range(len(reversed_gcn_dims) - 1):
            out_channels = reversed_gcn_dims[i+1]
            
            # Upsampling layer: halves the number of channels
            self.up_convs.append(
                nn.ConvTranspose2d(reversed_gcn_dims[i], out_channels, kernel_size=(1,2), stride=(1,2))
            )
            
            # Decoder block: input channels are doubled (from skip connection + upsample)
            # The output channels match the corresponding encoder level
            self.decoders.append(
                ConvGCNBlock(in_dim=out_channels * 2, out_dim=out_channels, K=K)
            )
            
        # --- Final layers from your original code ---
        self.manifold = PoincareBall(c=Curvature(requires_grad=True))
        
        if dist == 'poincare':
            self.linear = nn.Sequential(
                hnn.HLinear(gcn_dims[0], gcn_dims[0], manifold = self.manifold), 
                hnn.HReLU(manifold = self.manifold),
                hnn.HLinear(gcn_dims[0], gcn_dims[0], manifold = self.manifold), 
                hnn.HReLU(manifold = self.manifold),
                hnn.HLinear(gcn_dims[0], embedding_dim, manifold = self.manifold)
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(gcn_dims[0], gcn_dims[0]), nn.LeakyReLU(), 
                nn.Linear(gcn_dims[0], gcn_dims[0]), nn.LeakyReLU(),
                nn.Linear(gcn_dims[0], embedding_dim)
            )

        self.lstm = BiLSTM(gcn_dims[0], gcn_dims[0] // 2)
        
        self.act = nn.ELU()
        self.sig = nn.Sigmoid()
        self.eps = eps
        
        # --- NEW: Recombination Distance Projection Head ---
        if self.return_rec_dist:
            # Takes the pooled (gcn_dims[0]) vector and outputs a single scalar prediction
            self.rec_head = nn.Linear(gcn_dims[0], 1)
        
    def forward(self, x, use_l2 = False):        
            
        # --- Encoder Path ---
        skip_connections = []

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            
            if i < len(self.encoders) - 1:
                skip_connections.append(x)
                x = self.down_convs[i](x)

        skip_connections = skip_connections[::-1] 

        for i in range(len(self.up_convs)):
            x = self.up_convs[i](x)
            skip = skip_connections[i]

            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])

            concat_x = torch.cat((skip, x), dim=1) 
            x = self.decoders[i](concat_x)

        # --- Final Processing ---
        B, C, N, L = x.shape
        
        x = x.transpose(3, 1)
        xv = x.reshape(-1, N, C)
        xv = self.lstm(xv)
        
        # --------------------------------------------------------
        # NEW REGIME: Pool over N and predict single recombination distance
        # --------------------------------------------------------
        if self.return_rec_dist:
            xv_flat = xv.reshape(-1, C)
            
            # 1. Pass through all but the last layer `self.linear[:-1]`
            # This captures output immediately after the second-to-last linear layer (and its activation)
            if self.dist == "poincare":
                manifold_in = to_manifold(xv_flat, self.manifold)
                hidden_manifold = self.linear[:-1](manifold_in)
                hidden = hidden_manifold.tensor.to(torch.float32)
                
                # Still generate final embeddings just to return them correctly
                x_emb_raw = self.linear[-1](hidden_manifold).tensor.to(torch.float64)
                x_embedding = x_emb_raw.reshape(B, L, N, -1).to(torch.float32)
            else:
                hidden = self.linear[:-1](xv_flat)
                x_emb_raw = self.linear[-1](hidden)
                x_embedding = x_emb_raw.reshape(B, L, N, -1)
                
            # 2. Reshape and mean over N dimension 
            hidden = hidden.reshape(B, L, N, -1)
            pooled = hidden.mean(dim=2)  # Shape: (B, L, gcn_dims[0])
            
            # 3. Project pooled vector to a single distance prediction
            D = self.rec_head(pooled)    # Shape: (B, L, 1)
            
            return D, x_embedding
            
        # --------------------------------------------------------
        # OLD REGIME: Standard pairwise metric calculations
        # --------------------------------------------------------
        if self.dist == "l1":
            x_embedding = self.linear(xv.reshape(-1, C))
            x_embedding = x_embedding.reshape(B * L, N, -1)
            D = torch.log(Pdist(x_embedding) + self.eps)
            
        elif self.dist == "l2":
            x_embedding = self.linear(xv.reshape(-1, C))
            x_embedding = x_embedding.reshape(B * L, N, -1)
            D = torch.log(PdistL2(x_embedding)) * self.alpha + self.beta
            
        elif self.dist == "poincare":
            x_embedding = self.linear(to_manifold(xv.reshape(-1, C), self.manifold)).tensor.to(torch.float64)
            x_embedding = x_embedding.reshape(B * L, N, -1)
            D = torch.log(PdistPoincare(x_embedding)).to(torch.float32)
            
        else: # Lorentz
            raw_tangent = self.linear(xv.reshape(-1, C))
            raw_tangent = torch.renorm(raw_tangent, p=2, dim=0, maxnorm=15.0)
            tangent_f64 = raw_tangent.to(torch.float64)
            
            x_embedding = tangent_to_lorentz(tangent_f64.reshape(B * L, N, -1))
            D_raw = PdistLorentz(x_embedding, eps=1e-8)
            D = (torch.log(D_raw) * self.alpha + self.beta).to(torch.float32)

        out_dim = x_embedding.shape[-1]
        x_embedding = x_embedding.reshape(B, L, N, out_dim)
        D = D.reshape(B, L, *D.shape[1:])
                        
        return D, x_embedding.to(torch.float32)