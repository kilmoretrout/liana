use candle_core::{Device, DType, IndexOp, Result, Tensor};
use ndarray::{Array1, Array2};
use crate::layers::ConvGCNUNet;
use crate::wavelets::{cwt_real, RealWavelet}; 

pub struct UNetRegressor {
    pub model: ConvGCNUNet,
    pub scales: Vec<f32>,
    pub wavelet: RealWavelet,
    
    pub mu_x: Tensor,
    pub std_x: Tensor,
    pub mu_y: f64,
    pub std_y: f64,
    
    pub size: usize,
    pub stride: usize,
    pub device: Device,
}

impl UNetRegressor {
    pub fn new(
        model: ConvGCNUNet,
        scales: Vec<f32>,
        wavelet: RealWavelet,
        mu_x: Tensor,
        std_x: Tensor,
        mu_y: f64,
        std_y: f64,
        size: usize,
        stride: usize,
        device: Device,
    ) -> Self {
        Self {
            model, scales, wavelet, mu_x, std_x, mu_y, std_y, size, stride, device
        }
    }
    
    pub fn predict(
        &self, 
        x_ndarray: &Array2<f32>, 
        pos_ndarray: &Array1<f32>, 
        mask_channel: Option<usize>
    ) -> Result<Array2<f32>> {
        let n_individuals = x_ndarray.nrows();
        let sequence_length = x_ndarray.ncols();

        // ---------------------------------------------------------
        // 1. Compute CWT & Normalize (CPU -> GPU)
        // ---------------------------------------------------------
        let xw_ndarray = cwt_real(x_ndarray, &self.scales, self.wavelet.clone(), None);
        let xw_shape = xw_ndarray.dim();
        let xw_flat = xw_ndarray.into_raw_vec();
        let mut xw = Tensor::from_vec(xw_flat, (xw_shape.0, xw_shape.1, xw_shape.2), &self.device)?;
        
        let num_scales = xw_shape.0;
        let mu_x = self.mu_x.reshape((num_scales, 1, 1))?.to_device(&self.device)?;
        let std_x = self.std_x.reshape((num_scales, 1, 1))?.to_device(&self.device)?;
        
        xw = xw.broadcast_sub(&mu_x)?.broadcast_div(&std_x)?;

        // ---------------------------------------------------------
        // 2. Prepare Alignment (X) and Position (pos) Tensors
        // ---------------------------------------------------------
        let x_flat = x_ndarray.as_slice().unwrap().to_vec();
        let x_tensor = Tensor::from_vec(x_flat, (1, n_individuals, sequence_length), &self.device)?;

        let mut pos_diff = vec![0.0f32; sequence_length];
        pos_diff[0] = pos_ndarray[0];
        for i in 1..sequence_length {
            pos_diff[i] = pos_ndarray[i] - pos_ndarray[i - 1];
        }
        
        let pos_tensor = Tensor::from_vec(pos_diff, (1, 1, sequence_length), &self.device)?
            .broadcast_as((1, n_individuals, sequence_length))?;

        // Stack along Channel dimension -> Shape: (Channels, N, L)
        let mut x_input = Tensor::cat(&[&x_tensor, &xw, &pos_tensor], 0)?;

        if let Some(ch) = mask_channel {
            let total_channels = 2 + num_scales;
            let mut mask_vec = vec![1.0f32; total_channels];
            mask_vec[ch] = 0.0;
            let mask = Tensor::from_vec(mask_vec, (total_channels, 1, 1), &self.device)?;
            x_input = x_input.broadcast_mul(&mask)?;
        }

        // ---------------------------------------------------------
        // 3. Sliding Window Definitions
        // ---------------------------------------------------------
        let mut windows = Vec::new();
        let mut start = 0;
        let mut end = self.size;
        
        while end < sequence_length {
            windows.push((start, end));
            start += self.stride;
            end += self.stride;
        }
        
        // Append final window covering the tail edge
        if sequence_length > self.size {
            windows.push((sequence_length - self.size, sequence_length));
        } else {
            // Edge case: sequence is smaller than window size
            windows.push((0, sequence_length));
        }

        // ---------------------------------------------------------
        // 4. Inference & Accumulation
        // ---------------------------------------------------------
        let num_pairs = (n_individuals * (n_individuals - 1)) / 2;
        let mut d_pred = Array2::<f32>::zeros((sequence_length, num_pairs));
        let mut count = Array2::<f32>::zeros((sequence_length, num_pairs));

        for (a, b) in windows {
            let window_len = b - a;
            
            // Slice sequence: (C, N, L) -> (C, N, window_len)
            let xw_window = x_input.narrow(2, a, window_len)?;
            
            // Add Batch Dimension: (1, C, N, window_len)
            let xw_window = xw_window.unsqueeze(0)?;

            // Forward Pass
            let (dist_pred, _xe) = self.model.forward(&xw_window)?;
            
            // dist_pred shape is (Batch, SequenceLength, NumPairs) -> (1, window_len, num_pairs)
            // Squeeze batch dimension, move to CPU
            let dist_pred_cpu = dist_pred.squeeze(0)?.to_device(&Device::Cpu)?;

            // Math: exp(D * std_y + mu_y)
            // Candle's `affine` does `x * mul + add` optimally in C/CUDA
            let dist_pred_scaled = dist_pred_cpu
                .affine(self.std_y, self.mu_y)?
                .exp()?;
            
            // Flatten to 1D Vec for fast iterative assignment in Rust
            let dist_flat = dist_pred_scaled.flatten_all()?.to_vec1::<f32>()?;

            // Accumulate in standard NDArray
            for i in 0..window_len {
                for p in 0..num_pairs {
                    let val = dist_flat[i * num_pairs + p];
                    d_pred[[a + i, p]] += val;
                    count[[a + i, p]] += 1.0;
                }
            }
        }

        // ---------------------------------------------------------
        // 5. Final Averaging
        // ---------------------------------------------------------
        // Divide d_pred by count in place. If count is 0, we avoid div-by-zero NaN.
        d_pred.zip_mut_with(&count, |d, c| {
            if *c > 0.0 {
                *d /= *c;
            }
        });

        Ok(d_pred)
    }
}