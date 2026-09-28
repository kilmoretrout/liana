use candle_core::{Device, DType, IndexOp, Result, Tensor, D, D::Minus1};
use candle_nn::{Conv1d, Conv1dConfig, GroupNorm, LayerNorm, Linear, VarBuilder};
use candle_core::Module;
use candle_nn::RNN;

pub struct UpConv {
    conv: candle_nn::ConvTranspose1d,
    out_c: usize,
}

impl UpConv {
    fn new(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        // Set up the transposed convolution just like PyTorch: kernel_size=2, stride=2
        let cfg = candle_nn::ConvTranspose1dConfig {
            stride: 2,
            padding: 0,
            ..Default::default()
        };
        
        // Pass `vb` DIRECTLY (no .pp("proj")) so it looks for `up_convs.0.weight`.
        // If your PyTorch code actually nested it inside a variable (e.g. self.up = ...),
        // then you would use vb.pp("up") here. 
        let conv = candle_nn::conv_transpose1d(in_c, out_c, 2, cfg, vb)?;
        
        Ok(Self { conv, out_c }) // Wait, out_c might not even be needed anymore!
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x shape: (B, C, S)
        // Native ConvTranspose1d handles the upsampling automatically!
        self.conv.forward(x)
    }
}

// ---------------------------------------------------------
// Helper 1: Dense K-NN Batch Gather
// Replaces PyG's sparse edge_index scatter/gather.
// ---------------------------------------------------------
fn gather_neighbors(x: &Tensor, local_indices: &Tensor) -> Result<Tensor> {
    let (b, n, c) = x.dims3()?;
    let k = local_indices.dim(2)?;
    let device = x.device();
    
    // Create batch offsets: [0, N, 2N, ..., (B-1)N]
    let offsets: Vec<u32> = (0..b as u32).map(|i| i * n as u32).collect();
    let offsets = Tensor::new(offsets.as_slice(), device)?.reshape((b, 1, 1))?;
    
    // Convert local indices to global flattened indices and flatten the tensor
    let global_indices = local_indices.broadcast_add(&offsets)?.flatten_all()?;
    let flat_x = x.reshape((b * n, c))?;
    
    // Gather (B*N*K, C) and reshape back to (B, N, K, C)
    let gathered = flat_x.index_select(&global_indices, 0)?;
    gathered.reshape((b, n, k, c))
}

// ---------------------------------------------------------
// Helper 2: 1D Conv Wrapper for Asymmetric 2D Tensors
// ---------------------------------------------------------
fn apply_conv_norm(conv: &Conv1d, norm: Option<&GroupNorm>, inp: &Tensor) -> Result<Tensor> {
    let (b, c, n, s) = inp.dims4()?;
    
    // 1. (B, C, N, S) -> (B, N, C, S) -> (B*N, C, S)
    let flat_in = inp.transpose(1, 2)?.contiguous()?.reshape((b * n, c, s))?;
    let out_conv = conv.forward(&flat_in)?;
    
    let out_c = out_conv.dim(1)?;
    let out_s = out_conv.dim(2)?;
    
    // 2. (B*N, OutC, OutS) -> (B, N, OutC, OutS) -> (B, OutC, N, OutS)
    let mut out = out_conv.reshape((b, n, out_c, out_s))?.transpose(1, 2)?.contiguous()?;
    
    // 3. Apply norm on 4D tensor (acts as InstanceNorm2d)
    if let Some(n_layer) = norm {
        out = n_layer.forward(&out)?;
    }
    Ok(out)
}

// ---------------------------------------------------------
// Module: ResConvBlock & SingleConv
// ---------------------------------------------------------
struct ResConvBlock {
    conv0: Conv1d,
    norm0: GroupNorm,
    conv1: Conv1d,
    norm1: GroupNorm,
    conv2: Conv1d,
}

impl ResConvBlock {
    fn new(in_dim: usize, out_dim: usize, kernel_w: usize, dilations: [usize; 3], vb: VarBuilder) -> Result<Self> {
        let build_conv = |in_c, out_c, d, name| {
            let cfg = candle_nn::Conv1dConfig { padding: ((kernel_w - 1) * d) / 2, dilation: d, ..Default::default() };
            candle_nn::conv1d(in_c, out_c, kernel_w, cfg, vb.pp(name))
        };

        // PyTorch InstanceNorm1d defaults to affine=False (no saved weights).
        // We manually create a GroupNorm with 1.0s and 0.0s to bypass VarBuilder.
        let build_norm = |c: usize, _name: &str| -> Result<candle_nn::GroupNorm> {
            let weight = candle_core::Tensor::ones(c, candle_core::DType::F32, vb.device())?;
            let bias = candle_core::Tensor::zeros(c, candle_core::DType::F32, vb.device())?;
            // GroupNorm::new(weight, bias, num_channels, num_groups, eps)
            candle_nn::GroupNorm::new(weight, bias, c, c, 1e-5)
        };

        Ok(Self {
            conv0: build_conv(in_dim, out_dim, dilations[0], "conv0")?,
            norm0: build_norm(out_dim, "norm0")?,
            conv1: build_conv(out_dim, out_dim, dilations[1], "conv1")?,
            norm1: build_norm(out_dim, "norm1")?,
            conv2: build_conv(out_dim, out_dim, dilations[2], "conv2")?,
        })
    }

    fn forward(&self, x: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let x0 = apply_conv_norm(&self.conv0, Some(&self.norm0), x)?;
        let x0_act = candle_nn::ops::leaky_relu(&x0, 0.01)?; // PyTorch default LeakyReLU is 0.01

        let x1 = apply_conv_norm(&self.conv1, Some(&self.norm1), &x0_act)?;
        let x1_act = candle_nn::ops::leaky_relu(&x1, 0.01)?;

        // Skip connection: x_input (which is x0_act in the PyTorch code) + x1_act
        let skip_add = (&x0_act + &x1_act)?;
        
        apply_conv_norm(&self.conv2, None, &skip_add)
    }
}

// ---------------------------------------------------------
// Module: DenseGATv2Conv
// ---------------------------------------------------------
struct DenseGATv2Conv {
    lin_l: Linear,
    lin_r: Linear,
    att: Tensor,
    bias: Tensor,
    heads: usize,
    out_channels: usize,
}

impl DenseGATv2Conv {
    fn new(in_c: usize, out_c: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        // 1. Load the exact 3D shape PyTorch saved
        let att_raw = vb.get((1, heads, out_c), "att")?;
        
        // 2. Reshape it to 5D to match your forward pass math
        let att = att_raw.reshape((1, 1, 1, heads, out_c))?;

        Ok(Self {
            lin_l: candle_nn::linear(in_c, heads * out_c, vb.pp("lin_l"))?,
            lin_r: candle_nn::linear(in_c, heads * out_c, vb.pp("lin_r"))?,
            att,
            bias: vb.get(heads * out_c, "bias")?,
            heads,
            out_channels: out_c,
        })
    }

    fn forward(&self, x: &Tensor, knn_indices: &Tensor) -> Result<Tensor> {
        let (b, n, _c) = x.dims3()?;
        let k = knn_indices.dim(2)?;
        let h = self.heads;
        let out_c = self.out_channels;

        let x_l = self.lin_l.forward(x)?.reshape((b, n, 1, h, out_c))?;
        
        let x_src_raw = gather_neighbors(x, knn_indices)?;
        let x_r = self.lin_r.forward(&x_src_raw)?.reshape((b, n, k, h, out_c))?;

        let combined = candle_nn::ops::leaky_relu(&x_l.broadcast_add(&x_r)?, 0.2)?;
        
        // Compute Attention
        let alpha_raw = combined.broadcast_mul(&self.att)?.sum_keepdim(Minus1)?;
        let alpha = candle_nn::ops::softmax(&alpha_raw, 2)?; 

        // Aggregate Messages
        let msg = x_r.broadcast_mul(&alpha)?.sum(2)?;
        let out = msg.reshape((b, n, h * out_c))?;

        out.broadcast_add(&self.bias)
    }
}

// ---------------------------------------------------------
// Module: GCNBlock (with built-in distance and topk)
// ---------------------------------------------------------
struct GCNBlock {
    k: usize,
    gcns: Vec<DenseGATv2Conv>,
    norms: Vec<LayerNorm>,
}

impl GCNBlock {
    fn new(in_dim: usize, k: usize, num_layers: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        let mut gcns = Vec::new();
        let mut norms = Vec::new();
        
        for i in 0..num_layers {
            // In GatV2, out_channels is per-head. Total output is out_channels * heads
            // So we pass in_dim / heads as out_channels to maintain the dimension
            gcns.push(DenseGATv2Conv::new(in_dim, in_dim / heads, heads, vb.pp(format!("gcns.{}", i)))?);
            norms.push(candle_nn::layer_norm(in_dim, 1e-5, vb.pp(format!("norms.{}", i)))?);
        }
        Ok(Self { k, gcns, norms })
    }
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // 1. Compute Pairwise Euclidean Distance (equivalent to torch.cdist)
        let x1 = x.unsqueeze(2)?; // (B, N, 1, C)
        let x2 = x.unsqueeze(1)?; // (B, 1, N, C)
        let dist_sq = x1.broadcast_sub(&x2)?.sqr()?.sum(Minus1)?; // (B, N, N)
        
        // 2. K-NN Selection
        // Sort ascending. Because self-loop distance is 0, it is guaranteed to be at index 0.
        // We simply take neighbors 1 through K, skipping the 0th element.
        let sorted_indices = dist_sq.arg_sort_last_dim(true)?;
        let knn_indices = sorted_indices.narrow(2, 1, self.k)?; // (B, N, K)
        
        // 3. Message Passing
        let mut h = x.clone();
        for i in 0..self.gcns.len() {
            let out = self.gcns[i].forward(&h, &knn_indices)?;
            h = self.norms[i].forward(&(&h + out)?)?;
            if i < self.gcns.len() - 1 {
                h = candle_nn::ops::leaky_relu(&h, 0.01)?;
            }
        }
        
        Ok((h, knn_indices))
    }
}

// ---------------------------------------------------------
// Final Module: ConvGCNBlock 
// ---------------------------------------------------------
struct ConvGCNBlock {
    conv: ResConvBlock,
    gcn: GCNBlock,
}

impl ConvGCNBlock {
    fn new(in_dim: usize, out_dim: usize, k: usize, vb: VarBuilder) -> Result<Self> {
        // Add the extra .pp("conv") right here so it matches PyTorch's double nesting!
        let conv = ResConvBlock::new(in_dim, out_dim, 3, [1, 1, 1], vb.pp("conv").pp("conv"))?;
        
        // Defaulting to 2 layers and 4 heads like your Python code
        let gcn = GCNBlock::new(out_dim, k, 2, 4, vb.pp("gcn"))?; 
        
        Ok(Self { conv, gcn })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let (b, c, n, s) = x.dims4()?;
        
        // PyTorch: x.transpose(3, 1).reshape(-1, n, c)
        // Note: PyTorch dims are [0, 1, 2, 3], swapping 3 and 1 yields [B, S, N, C]
        let x_t = x.transpose(1, 3)?.contiguous()?;
        let x_gcn_in = x_t.reshape((b * s, n, c))?;
        
        // GCN Forward
        let (x_gcn_out, _edge_index) = self.gcn.forward(&x_gcn_in)?;
        
        // PyTorch: x.reshape(B, s, n, c).transpose(3, 1) -> maps back to [B, C, N, S]
        let x_final = x_gcn_out.reshape((b, s, n, c))?
                               .transpose(1, 3)?
                               .contiguous()?;
                               
        Ok(x_final)
    }
}

/// Helper function to mimic `torch.triu_indices(N, N, offset=1)`
/// Generates flattened 1D indices for the upper triangular part of an N x N matrix.
fn extract_triu(x: &Tensor) -> Result<Tensor> {
    let (b, n, _) = x.dims3()?;
    let num_elements = n * (n - 1) / 2;
    
    // Generate the flat indices for the upper triangle
    let mut indices = Vec::with_capacity(num_elements);
    for i in 0..n {
        for j in (i + 1)..n {
            indices.push((i * n + j) as u32);
        }
    }
    
    let indices_tensor = Tensor::new(indices.as_slice(), x.device())?;
    
    // Reshape to (B, N*N) and select the valid indices along dimension 1
    let flat_x = x.reshape((b, n * n))?;
    
    // Resulting shape: (B, N * (N - 1) / 2)
    flat_x.index_select(&indices_tensor, 1)
}

/// Computes pairwise L1 distances
fn pdist(x: &Tensor) -> Result<Tensor> {
    // x: (B, N, D)
    let x1 = x.unsqueeze(2)?; // (B, N, 1, D)
    let x2 = x.unsqueeze(1)?; // (B, 1, N, D)

    // Absolute difference and sum over the last dimension (D)
    // sum(3) removes the channel dimension, yielding (B, N, N)
    let diff = x1.broadcast_sub(&x2)?.abs()?.sum(3)?;
    
    extract_triu(&diff)
}

/// Computes pairwise L2 distances
fn pdist_l2(x: &Tensor) -> Result<Tensor> {
    // x: (B, N, D)
    let x1 = x.unsqueeze(2)?; // (B, N, 1, D)
    let x2 = x.unsqueeze(1)?; // (B, 1, N, D)

    // Squared difference sum + epsilon for stability
    let diff_sq = x1.broadcast_sub(&x2)?.sqr()?.sum(3)?;
    
    // Add epsilon and take sqrt: sqrt(sum((x1 - x2)^2) + 1e-12)
    let diff = (&diff_sq + 1e-12)?.sqrt()?;

    extract_triu(&diff)
}

/// Computes pairwise Poincaré distances
fn pdist_poincare(x: &Tensor, eps: f64) -> Result<Tensor> {
    // x: (B, N, D)
    let x1 = x.unsqueeze(2)?; // (B, N, 1, D)
    let x2 = x.unsqueeze(1)?; // (B, 1, N, D)

    // 1. Squared Euclidean distance: ||u - v||^2
    let euclidean_sq_dist = x1.broadcast_sub(&x2)?.sqr()?.sum(3)?; // (B, N, N)

    // 2. Compute squared norms and clamp: ||u||^2
    // We clamp the maximum to (1 - eps) to prevent div by zero
    let x_sq_norm = x.sqr()?.sum(D::Minus1)?; // (B, N)
    let x_sq_norm = x_sq_norm.clamp(f64::MIN, 1.0 - eps)?;

    // 3. Compute denominator term: (1 - ||u||^2) * (1 - ||v||^2)
    let norm_term = (1.0 - x_sq_norm)?; // (B, N)
    let norm_term_u = norm_term.unsqueeze(2)?; // (B, N, 1)
    let norm_term_v = norm_term.unsqueeze(1)?; // (B, 1, N)
    
    let denominator = norm_term_u.broadcast_mul(&norm_term_v)?;

    // 4. Poincaré formula argument: 1 + (2 * eucl_dist / denominator)
    let delta = ((&euclidean_sq_dist * 2.0)? / denominator)?;
    let x_val = (&delta + (1.0 + eps))?;

    // 5. Arccosh using log formula: log(x + sqrt(x^2 - 1))
    let x_val_sq_minus_one = (x_val.sqr()? - 1.0)?;
    let dist = (&x_val + x_val_sq_minus_one.sqrt()?)?.log()?;

    extract_triu(&dist)
}

/// Computes pairwise Lorentz distances
fn pdist_lorentz(x: &Tensor, eps: f64) -> Result<Tensor> {
    // x: (B, N, D)
    let d = x.dim(candle_core::D::Minus1)?;
    
    // 1. Split time and space components
    // narrow creates a non-contiguous view
    let x_time = x.narrow(2, 0, 1)?.squeeze(2)?; // (B, N)
    let x_space = x.narrow(2, 1, d - 1)?;        // (B, N, D-1)

    // 2. Pairwise Minkowski inner product
    
    // TIME PRODUCT:
    // Because we just want element-wise combinations of (B, N, 1) and (B, 1, N), 
    // we MUST use broadcast_mul instead of matmul. This ignores contiguity constraints!
    let x_time_u2 = x_time.unsqueeze(2)?;
    let x_time_u1 = x_time.unsqueeze(1)?;
    let time_prod = x_time_u2.broadcast_mul(&x_time_u1)?.neg()?; 
    
    // SPACE PRODUCT:
    // This requires a real MatMul, so we MUST force x_space to be contiguous first!
    let x_space = x_space.contiguous()?;
    let x_space_t = x_space.transpose(1, 2)?.contiguous()?; 
    let space_prod = x_space.matmul(&x_space_t)?; 

    // Total inner product
    let inner_prod = time_prod.broadcast_add(&space_prod)?;

    // 3. Stability check and Arccosh
    let safe_prod = inner_prod.neg()?.clamp(1.0 + eps, f64::MAX)?;
    
    let sq_minus_1 = (safe_prod.sqr()? - 1.0)?;
    let dist = (&safe_prod + sq_minus_1.sqrt()?)?.log()?;

    extract_triu(&dist)
}

use candle_nn::rnn::{lstm, LSTMConfig, LSTM};

// ---------------------------------------------------------
// Helper: Add Population Features
// ---------------------------------------------------------
fn add_pop_features(x: &Tensor) -> Result<Tensor> {
    let (b, c, n, l) = x.dims4()?;
    
    if n % 2 != 0 {
        return Err(candle_core::Error::Msg("N dimension must be even".to_string()));
    }
    
    let half_n = n / 2;
    
    // Create zeros for the first half, ones for the second half
    let zeros = Tensor::zeros((b, 1, half_n, l), x.dtype(), x.device())?;
    let ones = Tensor::ones((b, 1, half_n, l), x.dtype(), x.device())?;
    
    // Concatenate along the N dimension (dim = 2)
    let pop_channel = Tensor::cat(&[&zeros, &ones], 2)?;
    
    // Concatenate with original tensor along the Channel dimension (dim = 1)
    Tensor::cat(&[x, &pop_channel], 1)
}

// ---------------------------------------------------------
// Helper: Sequence Size Matching (replaces F.interpolate)
// ---------------------------------------------------------
fn match_sequence_length(x: &Tensor, target_len: usize) -> Result<Tensor> {
    let current_len = x.dim(3)?;
    if current_len == target_len {
        Ok(x.clone())
    } else if current_len > target_len {
        // Truncate the extra sequence elements
        x.narrow(3, 0, target_len)
    } else {
        // Pad with zeros (Constant padding)
        let diff = target_len - current_len;
        let pad = Tensor::zeros((x.dim(0)?, x.dim(1)?, x.dim(2)?, diff), x.dtype(), x.device())?;
        Tensor::cat(&[x, &pad], 3)
    }
}

// ---------------------------------------------------------
// Module: PopulationSMC_BiLSTM
// ---------------------------------------------------------
struct PopulationSMCBiLSTM {
    lstm_fwd_l0: LSTM,
    lstm_bwd_l0: LSTM,
    lstm_fwd_l1: LSTM,
    lstm_bwd_l1: LSTM,
    projection: candle_nn::Linear,
}

impl PopulationSMCBiLSTM {
    fn new(c_features: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = LSTMConfig::default();
        Ok(Self {
            // Layer 0 takes c_features
            lstm_fwd_l0: lstm(c_features, hidden_dim, cfg.clone(), vb.pp("lstm_fwd_l0"))?,
            lstm_bwd_l0: lstm(c_features, hidden_dim, cfg.clone(), vb.pp("lstm_bwd_l0"))?,
            
            // Layer 1 takes the concatenated output of Layer 0 (hidden_dim * 2)
            lstm_fwd_l1: lstm(hidden_dim * 2, hidden_dim, cfg.clone(), vb.pp("lstm_fwd_l1"))?,
            lstm_bwd_l1: lstm(hidden_dim * 2, hidden_dim, cfg.clone(), vb.pp("lstm_bwd_l1"))?,
            
            // The projection layer inside the LSTM module
            projection: candle_nn::linear(hidden_dim * 2, c_features, vb.pp("projection"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.dim(1)?; 
        
        // Helper closures for forward and backward sequences
        let apply_fwd = |layer: &LSTM, input: &Tensor| -> Result<Tensor> {
            let states = layer.seq(input)?;
            let h: Vec<Tensor> = states.into_iter().map(|s| s.h().clone()).collect();
            Tensor::stack(&h, 1)
        };
        
        let apply_bwd = |layer: &LSTM, input: &Tensor| -> Result<Tensor> {
            let rev_idx: Vec<u32> = (0..n as u32).rev().collect();
            let rev_tensor = Tensor::new(rev_idx.as_slice(), input.device())?;
            
            let x_rev = input.index_select(&rev_tensor, 1)?;
            let states = layer.seq(&x_rev)?;
            let h: Vec<Tensor> = states.into_iter().map(|s| s.h().clone()).collect();
            let h_stacked = Tensor::stack(&h, 1)?;
            
            h_stacked.index_select(&rev_tensor, 1) // reverse back
        };

        // --- Layer 0 ---
        let fwd0 = apply_fwd(&self.lstm_fwd_l0, x)?;
        let bwd0 = apply_bwd(&self.lstm_bwd_l0, x)?;
        let combined0 = Tensor::cat(&[&fwd0, &bwd0], 2)?;

        // --- Layer 1 ---
        let fwd1 = apply_fwd(&self.lstm_fwd_l1, &combined0)?;
        let bwd1 = apply_bwd(&self.lstm_bwd_l1, &combined0)?;
        let combined1 = Tensor::cat(&[&fwd1, &bwd1], 2)?;

        self.projection.forward(&combined1)
    }
}

fn tangent_to_lorentz(v: &Tensor, eps: f64) -> Result<Tensor> {
    // v shape: (B, N, D-1)
    
    // r is the Euclidean norm of the tangent vector
    let r = v.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sqrt()?; // (B, N, 1)
    
    // Compute exp(r) and exp(-r) once
    let exp_r = r.exp()?;
    let exp_neg_r = r.neg()?.exp()?;
    
    // Time component: cosh(r) = (exp(r) + exp(-r)) / 2
    // We use .affine(0.5, 0.0) which does `x * 0.5 + 0.0` extremely fast in C/CUDA
    let x_time = (&exp_r + &exp_neg_r)?.affine(0.5, 0.0)?;
    
    // Prevent division by zero
    let r_safe = r.clamp(eps, f64::MAX)?;
    
    // Space component: sinh(r) * (v / r_safe)
    let r_sinh = (&exp_r - &exp_neg_r)?.affine(0.5, 0.0)?;
    let v_normed = v.broadcast_div(&r_safe)?;
    let mut x_space = v_normed.broadcast_mul(&r_sinh)?;
    
    // Torch's where(r < eps, 0, x_space) via a 1.0/0.0 mask
    let mask = r.ge(eps)?.to_dtype(v.dtype())?; 
    x_space = x_space.broadcast_mul(&mask)?;
    
    // Concatenate along the last dimension
    Tensor::cat(&[&x_time, &x_space], candle_core::D::Minus1)
}

// ---------------------------------------------------------
// Main Module: ConvGCNUNet
// ---------------------------------------------------------
pub struct ConvGCNUNet {
    encoders: Vec<ConvGCNBlock>,
    down_convs: Vec<Conv1d>,
    decoders: Vec<ConvGCNBlock>,
    up_convs: Vec<UpConv>, 
    lstm: PopulationSMCBiLSTM,
    linear0: candle_nn::Linear, 
    linear2: candle_nn::Linear,
    linear4: candle_nn::Linear,
    alpha: Tensor,
    beta: Tensor,
    two_pop: bool,
    dist: String,
    eps: f64,
}

impl ConvGCNUNet {
    pub fn new(
        mut in_dim: usize, 
        gcn_dims: &[usize], 
        k: usize, 
        embedding_dim: usize, 
        two_pop: bool, 
        dist: String, 
        vb: VarBuilder
    ) -> Result<Self> {
        if two_pop {
            in_dim += 1; // Accommodate the binary population channel
        }
        
        let mut encoders = Vec::new();
        let mut down_convs = Vec::new();
        let mut curr_in = in_dim;
        
        // --- Encoders ---
        for (i, &out_dim) in gcn_dims.iter().enumerate() {
            // Use native .pp("encoders").pp(i) instead of format!
            encoders.push(ConvGCNBlock::new(curr_in, out_dim, k, vb.pp("encoders").pp(i))?);
            curr_in = out_dim;
            
            // Add downsample convolution (except for the bottleneck layer)
            if i < gcn_dims.len() - 1 {
                let cfg = candle_nn::Conv1dConfig { stride: 2, padding: 0, ..Default::default() };
                down_convs.push(candle_nn::conv1d(
                    out_dim, out_dim, 2, cfg, vb.pp("down_convs").pp(i)
                )?);
            }
        }
        
        let mut up_convs = Vec::new();
        let mut decoders = Vec::new();
        
        // --- Decoders ---
        let reversed_dims: Vec<usize> = gcn_dims.iter().copied().rev().collect();
        for i in 0..reversed_dims.len() - 1 {
            let out_channels = reversed_dims[i + 1];
            up_convs.push(UpConv::new(
                reversed_dims[i], out_channels, vb.pp("up_convs").pp(i)
            )?);
            
            // Multiply input by 2 to account for skip connection concatenation
            decoders.push(ConvGCNBlock::new(
                out_channels * 2, out_channels, k, vb.pp("decoders").pp(i)
            )?);
        }
        
        // --- Final Layers ---
        let lstm = PopulationSMCBiLSTM::new(gcn_dims[0], gcn_dims[0] / 2, vb.pp("lstm"))?;
        
        let mlp_hidden = gcn_dims[0]; 
        let linear0 = candle_nn::linear(gcn_dims[0], mlp_hidden, vb.pp("linear").pp("0"))?;
        let linear2 = candle_nn::linear(mlp_hidden, mlp_hidden, vb.pp("linear").pp("2"))?;
        let linear4 = candle_nn::linear(mlp_hidden, embedding_dim, vb.pp("linear").pp("4"))?;
        
        let alpha = vb.get(1, "alpha")?;
        let beta = vb.get(1, "beta")?;
        
        Ok(Self {
            encoders, down_convs, decoders, up_convs, lstm, linear0, linear2, linear4, alpha, beta, two_pop, dist, eps: 1e-8
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut h = if self.two_pop {
            add_pop_features(x)?
        } else {
            x.clone()
        };

        // --- Encoder Path ---
        let mut skip_connections = Vec::new();
        
        for i in 0..self.encoders.len() {
            h = self.encoders[i].forward(&h)?;
            
            if i < self.encoders.len() - 1 {
                skip_connections.push(h.clone());
                
                // Downsample using 1D trick over flattened B*N
                let (b, c, n, s) = h.dims4()?;
                let flat_in = h.transpose(1, 2)?.contiguous()?.reshape((b * n, c, s))?;
                let flat_out = self.down_convs[i].forward(&flat_in)?;
                
                let out_s = flat_out.dim(2)?;
                h = flat_out.reshape((b, n, c, out_s))?.transpose(1, 2)?.contiguous()?;
            }
        }

        // --- Decoder Path ---
        // Pop skip connections in reverse order naturally
        for i in 0..self.up_convs.len() {
            // Upsample
            let (b, c, n, s) = h.dims4()?;
            let flat_in = h.transpose(1, 2)?.contiguous()?.reshape((b * n, c, s))?;
            
            // Forward through our custom UpConv layer
            let flat_out = self.up_convs[i].forward(&flat_in)?;
            
            let out_c = flat_out.dim(1)?;
            let out_s = flat_out.dim(2)?;
            h = flat_out.reshape((b, n, out_c, out_s))?.transpose(1, 2)?.contiguous()?;
            
            let skip = skip_connections.pop().unwrap();
            h = match_sequence_length(&h, skip.dim(3)?)?;
            
            let concat_h = Tensor::cat(&[&skip, &h], 1)?;
            h = self.decoders[i].forward(&concat_h)?;
        }

        // --- Final Processing ---
        let (b, c, n, l) = h.dims4()?; // <--- Keep this as `c`
        
        // Transpose to (B, L, N, C) and flatten to (B*L, N, C)
        let xv = h.transpose(1, 3)?.contiguous()?.reshape((b * l, n, c))?;
        
        // LSTM Pass
        let xv = self.lstm.forward(&xv)?;
        
        // Linear Pass
        let x_flat = xv.reshape((b * l * n, c))?;
        
        // DELETE: let raw_tangent = self.linear.forward(&x_flat)?;
        // ADD THESE:
        let x_h1 = candle_nn::ops::leaky_relu(&self.linear0.forward(&x_flat)?, 0.01)?;
        let x_h2 = candle_nn::ops::leaky_relu(&self.linear2.forward(&x_h1)?, 0.01)?;
        let raw_tangent = self.linear4.forward(&x_h2)?;
        
        // PyTorch renorm(p=2, dim=0, maxnorm=15.0)
        // Computes the L2 norm for each row (dim 1), keeping the dimension so we can broadcast
        let norm = raw_tangent.sqr()?.sum_keepdim(1)?.sqrt()?;
        // Factor = min(1.0, 15.0 / max(norm, 1e-9))
        let safe_norm = norm.clamp(1e-9, f64::MAX)?;
        let factor = (15.0 / &safe_norm)?.clamp(0.0, 1.0)?;
        let raw_tangent = raw_tangent.broadcast_mul(&factor)?;
        
        let out_dim = raw_tangent.dim(1)?;
        
        // Cast to f64 for stable hyperbolic geometry math
        let tangent_f64 = raw_tangent.to_dtype(DType::F64)?;
        let mut x_embedding = tangent_f64.reshape((b * l, n, out_dim))?;

        // Distance Calculation
        let mut dist = match self.dist.as_str() {
            "l1" => pdist(&x_embedding)?,
            "l2" => pdist_l2(&x_embedding)?,
            "poincare" => {
                pdist_poincare(&x_embedding, self.eps)?
            }
            "lorentz" => {
                x_embedding = tangent_to_lorentz(&x_embedding, 1e-9)?;
                pdist_lorentz(&x_embedding, self.eps)?
            }
            _ => pdist_l2(&x_embedding)?,
        };

        // log(D)
        dist = dist.log()?;
        
        // Cast back to f32 BEFORE applying alpha and beta (which are f32 Tensors)
        dist = dist.to_dtype(DType::F32)?;
        x_embedding = x_embedding.to_dtype(DType::F32)?;
        
        if self.dist != "poincare" { 
            let a = self.alpha.broadcast_as(dist.shape())?;
            let b_tensor = self.beta.broadcast_as(dist.shape())?;
            dist = dist.broadcast_mul(&a)?.broadcast_add(&b_tensor)?;
        }

        // Reshape outputs back to (B, L, ...)
        let emb_dim = x_embedding.dim(candle_core::D::Minus1)?;
        x_embedding = x_embedding.reshape((b, l, n, emb_dim))?;
        
        let num_pairs = dist.dim(1)?;
        dist = dist.reshape((b, l, num_pairs))?;

        Ok((dist, x_embedding))
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Import everything from layers.rs
    use candle_core::{Device, Tensor, DType};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn test_add_pop_features() -> Result<()> {
        let device = Device::Cpu;
        
        // (Batch=2, Channels=1, N=4, L=10)
        let x = Tensor::zeros((2, 1, 4, 10), DType::F32, &device)?;
        
        let out = add_pop_features(&x)?;
        
        // Channel dimension should increase by 1
        assert_eq!(out.dims4()?, (2, 2, 4, 10));
        
        // Check that the new population channel (index 1) is 0 for first half of N, 1 for second half
        let pop_channel = out.narrow(1, 1, 1)?; // Select the appended channel
        
        // Get an element from the first half (n=0..2) and second half (n=2..4)
        let val_first_half: f32 = pop_channel.i((0, 0, 1, 5))?.to_scalar()?;
        let val_second_half: f32 = pop_channel.i((0, 0, 3, 5))?.to_scalar()?;
        
        assert_eq!(val_first_half, 0.0);
        assert_eq!(val_second_half, 1.0);
        
        Ok(())
    }

    #[test]
    fn test_pdist_l2() -> Result<()> {
        let device = Device::Cpu;
        
        // Batch = 1, N = 3, D = 2
        // Point 0: (0.0, 0.0)
        // Point 1: (3.0, 4.0) -> dist to 0 is 5.0
        // Point 2: (0.0, 5.0) -> dist to 0 is 5.0, dist to 1 is sqrt(3^2 + 1^2) = 3.1622
        let data: Vec<f32> = vec![
            0.0, 0.0, 
            3.0, 4.0, 
            0.0, 5.0
        ];
        
        let x = Tensor::from_vec(data, (1, 3, 2), &device)?;
        
        let dists = pdist_l2(&x)?;
        
        // N=3 means N*(N-1)/2 = 3 pairwise distances.
        // Expected order for triu offset=1: (0,1), (0,2), (1,2)
        assert_eq!(dists.dims2()?, (1, 3));
        
        let dist_vec: Vec<f32> = dists.flatten_all()?.to_vec1()?;
        
        // Allow a small epsilon for floating point math
        assert!((dist_vec[0] - 5.0).abs() < 1e-4);
        assert!((dist_vec[1] - 5.0).abs() < 1e-4);
        assert!((dist_vec[2] - 3.162277).abs() < 1e-4);
        
        Ok(())
    }

    #[test]
    fn test_bilstm_shape() -> Result<()> {
        let device = Device::Cpu;
        
        // To initialize network layers without loading real weights, 
        // we use a VarMap populated with zeros.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let c_features = 64;
        let hidden_dim = 32;
        
        let bilstm = PopulationSMCBiLSTM::new(c_features, hidden_dim, vb)?;
        
        // Input: (B*L, N, C) -> e.g., (10, 4, 64)
        let input = Tensor::zeros((10, 4, 64), DType::F32, &device)?;
        
        let output = bilstm.forward(&input)?;
        
        // Output should project back to the original c_features dimension
        assert_eq!(output.dims3()?, (10, 4, 64));
        
        Ok(())
    }
    
    #[test]
    fn test_full_conv_gcn_unet() -> Result<()> {
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        println!("Running test on device: {:?}", device);
        
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        // We'll bump the dimensions up slightly to give the GPU some actual work to do
        let in_dim = 108;
        let embedding_dim = 14;
        let gcn_dims = vec![512, 256, 128, 64]; 
        let k = 11; 
        
        let model = ConvGCNUNet::new(in_dim, &gcn_dims, k, embedding_dim, true, "lorentz".to_string(), vb)?;
        
        // Simulating a decent sized batch: 2 batches, 100 individuals, 64 sequence length
        let b = 1;
        let c = 108;
        let n = 32;
        let l = 512;  
        let x = Tensor::zeros((b, c, n, l), DType::F32, &device)?;
        
        println!("Running warm-up pass (allocating memory & compiling kernels)...");
        let _ = model.forward(&x)?;
        
        // MUST sync the device before starting the timer so the warm-up is 100% finished
        device.synchronize()?; 
        
        println!("Running timed forward pass...");
        let start = std::time::Instant::now();
        
        let (dist, embed) = model.forward(&x)?;
        
        // MUST sync the device again before stopping the timer to wait for the GPU to finish
        device.synchronize()?;
        let duration = start.elapsed();
        
        // Expected shape logic
        // N = 100 individuals -> (100 * 99) / 2 = 4950 pairs
        let expected_pairs = (n * (n - 1)) / 2;
        
        println!("--------------------------------------------------");
        println!("Output Distance Shape : {:?}", dist.shape());
        println!("Output Embedding Shape: {:?}", embed.shape());
        println!("Forward Pass Time     : {:.2?} (Batch: {}, N: {}, L: {})", duration, b, n, l);
        println!("--------------------------------------------------");
        
        Ok(())
    }
}