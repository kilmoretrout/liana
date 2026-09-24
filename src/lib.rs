use std::f64::consts::PI;

use pyo3::prelude::*;
use numpy::{PyArray2, PyArray3, ToPyArray};
use ndarray::Array2;
use ndarray::Array3;
use realfft::RealFftPlanner;
use ndarray::Axis;
use ndarray::parallel::prelude::IntoParallelIterator;
use ndarray::parallel::prelude::IndexedParallelIterator;
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::parallel::prelude::ParallelIterator;

// 1. Your internal Rust Enum
#[derive(Clone)]
pub enum RealWavelet {
    Gaussian { order: u32 },
    Shannon,
    Morlet { w0: f32 },
    Haar,
    Smc, // Added Custom SMC Wavelet
}

// 2. The Python-facing Bridge Class
#[pyclass]
#[derive(Clone)]
pub struct WaveletConfig {
    pub internal: RealWavelet,
}

#[pymethods]
impl WaveletConfig {
    #[staticmethod]
    fn gaussian(order: u32) -> Self {
        Self { internal: RealWavelet::Gaussian { order } }
    }

    #[staticmethod]
    fn shannon() -> Self {
        Self { internal: RealWavelet::Shannon }
    }

    #[staticmethod]
    fn morlet(w0: f32) -> Self {
        Self { internal: RealWavelet::Morlet { w0 } }
    }

    #[staticmethod]
    fn haar() -> Self {
        Self { internal: RealWavelet::Haar }
    }

    #[staticmethod]
    fn smc() -> Self {
        Self { internal: RealWavelet::Smc } // Added SMC constructor
    }
}

#[pyfunction]
#[pyo3(signature = (input, scales, config, pad_len=None))]
fn compute_cwt<'py>(
    py: Python<'py>,
    input: &PyArray2<f32>,
    scales: Vec<f32>,
    config: &WaveletConfig,
    pad_len: Option<usize>
) -> &'py PyArray3<f32> {
    let input_ndarray = unsafe { input.as_array() };
    
    // Call our auto-padding logic
    let result = cwt_real(
        &input_ndarray.to_owned(), 
        &scales, 
        config.internal.clone(),
        pad_len
    );
    
    result.to_pyarray(py)
}

#[pymodule]
fn pgml(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_cwt, m)?)?;
    m.add_class::<WaveletConfig>()?; 
    
    Ok(())
}

impl RealWavelet {
    pub fn generate(&self, len: usize, scale: f32) -> Vec<f32> {
        let center = len as f32 / 2.0f32;
        (0..len).map(|t| {
            let x = (t as f32 - center) / scale;
            match self {
                RealWavelet::Gaussian { order } => self.gaussian_derivative(x, *order),
                RealWavelet::Shannon => {
                    if x == 0.0 { 1.0 } else { ((PI as f32) * x).sin() / ((PI as f32) * x) }
                },
                RealWavelet::Morlet { w0 } => {
                    // Real part of Morlet: cos(w0 * x) * exp(-x^2 / 2)
                    (w0 * x).cos() * (-x * x / 2.0).exp()
                },
                RealWavelet::Haar => {
                    // Centered Haar wavelet
                    if x >= -0.5 && x < 0.0 {
                        1.0
                    } else if x >= 0.0 && x < 0.5 {
                        -1.0
                    } else {
                        0.0
                    }
                },
                RealWavelet::Smc => {
                    // First derivative of the double-exponential Laplace kernel
                    if x > 0.0 {
                        (-x).exp()
                    } else if x < 0.0 {
                        -(x.exp())
                    } else {
                        0.0
                    }
                }
            }
        }).collect()
    }

    // Computes the n-th derivative of exp(-x^2/2)
    fn gaussian_derivative(&self, x: f32, order: u32) -> f32 {
        let envelope = (-x * x / 2.0).exp();
        match order {
            0 => envelope, // Pure Gaussian (Smoothing)
            1 => -x * envelope, // First Derivative
            2 => (x * x - 1.0) * envelope, // Second Derivative (Ricker/Mexican Hat)
            3 => (3.0 * x - x.powi(3)) * envelope,
            4 => (x.powi(4) - 6.0 * x * x + 3.0) * envelope,
            _ => self.hermite_recursive(x, order) * envelope,
        }
    }

    // General case for high orders using the recurrence: 
    // H_{n+1}(x) = xH_n(x) - nH_{n-1}(x)
    fn hermite_recursive(&self, x: f32, n: u32) -> f32 {
        let mut h_prev = 0.0; // H_{-1} is effectively 0
        let mut h_curr = 1.0; // H_0
        
        // Note: For derivatives, we actually use the probabilistic Hermite polynomials
        // resulting in the relationship: He_{n+1} = xHe_n - nHe_{n-1}
        for i in 0..n {
            let h_next = x * h_curr - (i as f32) * h_prev;
            h_prev = h_curr;
            h_curr = h_next;
        }
        
        // The n-th derivative includes the (-1)^n factor
        if n % 2 == 1 { -h_curr } else { h_curr }
    }
}

fn next_power_of_2(n: usize) -> usize {
    if n == 0 { return 1; }
    let mut val = n;
    val -= 1;
    val |= val >> 1;
    val |= val >> 2;
    val |= val >> 4;
    val |= val >> 8;
    val |= val >> 16;
    #[cfg(target_pointer_width = "64")]
    { val |= val >> 32; }
    val + 1
}

pub fn cwt_real(
    input: &Array2<f32>, 
    scales: &[f32], 
    wavelet: RealWavelet,
    user_pad: Option<usize> // Let user override if they want
) -> Array3<f32> {
    let (n_signals, length) = input.dim();
    
    // Rule of thumb: pad by 4 * max_scale
    // We use .fold to find max in case scales isn't sorted
    let max_scale = scales.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let pad_len = user_pad.unwrap_or((max_scale * 4.0).ceil() as usize);
    
    let padded_length = next_power_of_2(length + 2 * pad_len);
    let mut output = Array3::<f32>::zeros((scales.len(), n_signals, length));
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(padded_length);
    let c2r = planner.plan_fft_inverse(padded_length);
    
    output.axis_iter_mut(Axis(0)).into_par_iter()
        .zip(scales.par_iter())
        .for_each(|(mut scale_slice, &scale)| {            
            // 2. Use the shared r2c and c2r plans inside the loop safely
            let mut kernel_fft = r2c.make_output_vec();
            r2c.process(&mut wavelet.generate(padded_length, scale), &mut kernel_fft).unwrap();

            scale_slice.axis_iter_mut(Axis(0)).into_par_iter().enumerate()
            .for_each_init(
                || {
                    // This closure runs exactly once per Rayon thread to allocate reusable memory
                    (
                        vec![0.0f32; padded_length],
                        r2c.make_output_vec(),
                        c2r.make_output_vec()
                    )
                },
                |(padded_signal, signal_fft, result_time), (row_idx, mut row)| {
                    
                    // Clear the padded signal buffer for the new row instead of reallocating
                    padded_signal.fill(0.0);
                    
                    let signal_row = input.row(row_idx);
                    for i in 0..length {
                        padded_signal[i + pad_len] = signal_row[i];
                    }
        
                    // The FFT process writes directly into your reused buffers
                    r2c.process(padded_signal, signal_fft).unwrap();
        
                    for i in 0..signal_fft.len() {
                        signal_fft[i] *= kernel_fft[i];
                    }
        
                    c2r.process(signal_fft, result_time).unwrap();
        
                    let norm = padded_length as f32 * scale.sqrt();
                    for i in 0..length {
                        row[i] = result_time[i + pad_len] / norm;
                    }
                }
            );
        });
    output
}