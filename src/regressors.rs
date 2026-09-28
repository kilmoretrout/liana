use candle_core::{Device, DType, IndexOp, Result, Tensor};
use ndarray::{Array1, Array2};
use crate::layers::ConvGCNUNet;
use crate::{cwt_real, RealWavelet}; 

use std::fs::File;
use std::io::{BufWriter, Write};
use ndarray::s;
use indicatif::{ProgressBar, ProgressStyle}; 

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
    

    pub fn predict_to_disk(
        &self, 
        x_ndarray: &Array2<f32>, 
        pos_ndarray: &Array1<f32>, 
        mask_channel: Option<usize>,
        total_base_pairs: f32,
        output_filepath: &str
    ) -> Result<()> {
        let n_individuals = x_ndarray.nrows();
        let num_snps = x_ndarray.ncols(); 
        let num_pairs = (n_individuals * (n_individuals - 1)) / 2;
        let num_scales = self.scales.len();
        
        let mu_x = self.mu_x.reshape((num_scales, 1, 1))?.to_device(&self.device)?;
        let std_x = self.std_x.reshape((num_scales, 1, 1))?.to_device(&self.device)?;

        let file = File::create(output_filepath)?;
        let mut writer = BufWriter::new(file);

        let window_size_bp = 1_000_000.0; 

        // ---------------------------------------------------------
        // 1. Fast Dry-Run to Count Exact Number of Windows
        // ---------------------------------------------------------
        let mut total_windows = 0;
        let mut temp_bp = 0.0;
        let mut temp_idx = 0;
        while temp_idx < num_snps {
            let target_bp = temp_bp + window_size_bp;
            let mut end_idx = temp_idx;
            while end_idx < num_snps && pos_ndarray[end_idx] < target_bp {
                end_idx += 1;
            }
            if end_idx == temp_idx {
                temp_bp = target_bp;
                continue;
            }
            let len = end_idx - temp_idx;
            let mut start = 0;
            while start < len {
                let end = usize::min(start + self.size, len);
                total_windows += 1;
                if end - start < self.size && len >= self.size {
                    break;
                }
                start += self.stride;
            }
            temp_bp = target_bp;
            temp_idx = end_idx;
        }

        // ---------------------------------------------------------
        // 2. Setup Progress Bar to track Windows
        // ---------------------------------------------------------
        let pb = ProgressBar::new(total_windows as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Windows ({eta}) | {msg}")
                .unwrap()
                .progress_chars("=>-")
        );
        pb.set_message("Starting inference...");

        // ---------------------------------------------------------
        // 3. Main Processing Loop
        // ---------------------------------------------------------
        let mut current_bp = 0.0;
        let mut chunk_start_idx = 0; 

        while chunk_start_idx < num_snps {
            let target_bp = current_bp + window_size_bp;
            
            let mut chunk_end_idx = chunk_start_idx;
            while chunk_end_idx < num_snps && pos_ndarray[chunk_end_idx] < target_bp {
                chunk_end_idx += 1;
            }
            
            if chunk_end_idx == chunk_start_idx {
                current_bp = target_bp;
                continue;
            }

            let current_chunk_len = chunk_end_idx - chunk_start_idx;
            let x_chunk = x_ndarray.slice(s![.., chunk_start_idx..chunk_end_idx]).to_owned();
            
            // Safe CWT padding (Guarantees Even Length for FFT)
            let xw_chunk_cpu = if current_chunk_len % 2 != 0 {
                let padding = ndarray::Array2::<f32>::zeros((n_individuals, 1));
                let x_chunk_padded = ndarray::concatenate(ndarray::Axis(1), &[x_chunk.view(), padding.view()]).unwrap();
                let xw_padded = cwt_real(&x_chunk_padded, &self.scales, self.wavelet.clone(), None);
                xw_padded.slice(s![.., .., ..current_chunk_len]).to_owned()
            } else {
                cwt_real(&x_chunk, &self.scales, self.wavelet.clone(), None)
            };

            let pos_chunk = pos_ndarray.slice(s![chunk_start_idx..chunk_end_idx]).to_vec();
            let mut pos_diff = vec![0.0f32; current_chunk_len];
            
            if chunk_start_idx == 0 {
                pos_diff[0] = pos_chunk[0] / window_size_bp;
            } else {
                pos_diff[0] = (pos_chunk[0] - pos_ndarray[chunk_start_idx - 1]) / window_size_bp; 
            }
            for i in 1..current_chunk_len {
                pos_diff[i] = (pos_chunk[i] - pos_chunk[i - 1]) / window_size_bp;
            }

            let mut d_pred = Array2::<f32>::zeros((current_chunk_len, num_pairs));
            let mut count = Array2::<f32>::zeros((current_chunk_len, num_pairs));

            // Safe UNet Windowing
            let mut start = 0;
            let mut windows = Vec::new();
            
            while start < current_chunk_len {
                let end = usize::min(start + self.size, current_chunk_len);
                if end - start < self.size && current_chunk_len >= self.size {
                    windows.push((current_chunk_len - self.size, current_chunk_len));
                    break;
                } else {
                    windows.push((start, end));
                }
                start += self.stride;
            }

            // --- INNER LOOP (Ticking the Progress Bar) ---
            for (a, b) in windows {
                let window_len = b - a;
                let pad_len = if window_len < self.size { self.size - window_len } else { 0 };

                // Update the terminal message dynamically for the current window being processed
                pb.set_message(format!(
                    "{:.1}-{:.1} Mb (SNPs {} to {})", 
                    current_bp / 1_000_000.0, 
                    target_bp / 1_000_000.0,
                    a, b
                ));

                let xw_win_cpu = xw_chunk_cpu.slice(s![.., .., a..b]);
                let xw_tensor = Tensor::from_vec(
                    xw_win_cpu.to_owned().into_raw_vec(), 
                    (num_scales, n_individuals, window_len), 
                    &self.device
                )?;
                let xw_tensor = xw_tensor.broadcast_sub(&mu_x)?.broadcast_div(&std_x)?;

                let x_win_cpu = x_chunk.slice(s![.., a..b]).to_owned().into_raw_vec();
                let x_tensor = Tensor::from_vec(x_win_cpu, (1, n_individuals, window_len), &self.device)?;

                let pos_win_cpu = pos_diff[a..b].to_vec();
                let pos_tensor = Tensor::from_vec(pos_win_cpu, (1, 1, window_len), &self.device)?
                    .broadcast_as((1, n_individuals, window_len))?;

                let mut x_input = Tensor::cat(&[&x_tensor, &xw_tensor, &pos_tensor], 0)?;

                if let Some(ch) = mask_channel {
                    let total_channels = 2 + num_scales;
                    let mut mask_vec = vec![1.0f32; total_channels];
                    mask_vec[ch] = 0.0;
                    let mask = Tensor::from_vec(mask_vec, (total_channels, 1, 1), &self.device)?;
                    x_input = x_input.broadcast_mul(&mask)?;
                }

                x_input = x_input.unsqueeze(0)?;

                if pad_len > 0 {
                    x_input = x_input.pad_with_zeros(3, 0, pad_len)?;
                }

                let (dist_pred_raw, _xe) = self.model.forward(&x_input)?;
                
                let dist_pred = if pad_len > 0 {
                    dist_pred_raw.narrow(1, 0, window_len)?
                } else {
                    dist_pred_raw
                };
                
                let dist_pred_cpu = dist_pred.squeeze(0)?.to_device(&Device::Cpu)?;
                let dist_pred_scaled = dist_pred_cpu.affine(self.std_y, self.mu_y)?.exp()?;
                let dist_flat = dist_pred_scaled.flatten_all()?.to_vec1::<f32>()?;

                for i in 0..window_len {
                    for p in 0..num_pairs {
                        let val = dist_flat[i * num_pairs + p];
                        d_pred[[a + i, p]] += val;
                        count[[a + i, p]] += 1.0;
                    }
                }

                // Increment progress bar by 1 window!
                pb.inc(1);
            } 

            d_pred.zip_mut_with(&count, |d, c| {
                if *c > 0.0 { *d /= *c; }
            });

            let raw_bytes = unsafe {
                std::slice::from_raw_parts(
                    d_pred.as_ptr() as *const u8,
                    d_pred.len() * std::mem::size_of::<f32>()
                )
            };
            
            writer.write_all(raw_bytes).expect("Failed to write to disk");
            writer.flush()?;

            current_bp = target_bp;
            chunk_start_idx = chunk_end_idx;
        }
        
        pb.finish_with_message("Inference complete and saved to disk!");
        Ok(())
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