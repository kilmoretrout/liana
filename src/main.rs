use clap::Parser;
use std::path::PathBuf;
use std::fs::File;
use std::io::{BufReader, Write};

use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use ndarray::{Array1, Array2};

use liana::layers::ConvGCNUNet;
use liana::regressors::UNetRegressor;
use liana::RealWavelet;
use noodles::vcf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the input VCF file (uncompressed or bgzipped)
    #[arg(short, long)]
    input: PathBuf,

    /// Path to output the tree sequence (.trees)
    #[arg(short, long)]
    output: PathBuf,

    /// Path to the model weights (.safetensors)
    #[arg(short, long)]
    model_weights: PathBuf,

    /// Enable CUDA acceleration
    #[arg(long, default_value_t = false)]
    cuda: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // 1. Setup Device
    let device = if args.cuda {
        Device::new_cuda(0).unwrap_or_else(|_| {
            eprintln!("Warning: CUDA requested but not found. Falling back to CPU.");
            Device::Cpu
        })
    } else {
        Device::Cpu
    };
    println!("Using device: {:?}", device);

    // ---------------------------------------------------------
    // 2. Parse VCF into NDArray
    // ---------------------------------------------------------
    println!("Parsing VCF: {:?}", args.input);
    
    // Use noodles strictly for blazing-fast decompression/IO
    let file = File::open(&args.input)?;
    let mut reader = noodles::bgzf::Reader::new(file);
    
    let mut line = String::new();
    let mut n_individuals = 0;
    
    // 1. Read header to get the number of individuals
    while reader.read_line(&mut line)? > 0 {
        if line.starts_with("#CHROM") {
            let cols: Vec<&str> = line.trim_end().split('\t').collect();
            // Genotypes start at column 9 (0-indexed)
            n_individuals = cols.len().saturating_sub(9);
            line.clear();
            break;
        }
        line.clear();
    }

    let mut pos_vec = Vec::new();
    let mut genotype_matrix = Vec::new(); 

    // 2. Parse variants
    while reader.read_line(&mut line)? > 0 {
        if line.starts_with('#') { 
            line.clear(); 
            continue; 
        }
        
        let cols: Vec<&str> = line.trim_end().split('\t').collect();
        if cols.len() < 9 { 
            line.clear(); 
            continue; 
        }
        
        // Extract Position (Column 1 is POS)
        let pos: f32 = cols[1].parse().expect("Failed to parse position");
        pos_vec.push(pos);

        // Extract Genotypes (Columns 9 to End)
        for i in 9..cols.len() {
            // The GT field is always the first sub-field before the colon
            let gt_str = cols[i].split(':').next().unwrap_or("0");
            
            // Simple biallelic parser: if it contains '1', code as 1.0
            let allele = if gt_str.contains('1') { 1.0f32 } else { 0.0f32 };
            genotype_matrix.push(allele);
        }
        
        line.clear();
    }

    let sequence_length = pos_vec.len();
    println!("Parsed {} variants for {} individuals.", sequence_length, n_individuals);

    // Reshape the flat vector into an (N, L) matrix
    let x_ndarray = Array2::from_shape_vec((n_individuals, sequence_length), genotype_matrix)?;
    let pos_ndarray = Array1::from_vec(pos_vec);

    // ---------------------------------------------------------
    // 3. Load Neural Network Weights
    // ---------------------------------------------------------
    println!("Loading model weights from {:?}", args.model_weights);
    let vb = unsafe { 
        VarBuilder::from_mmaped_safetensors(&[&args.model_weights], DType::F32, &device)? 
    };

    // Initialize Network (Hardcoded architectural params for now, you can move these to Args/Config)
    let in_dim = 130;
    let embedding_dim = 14;
    let gcn_dims = vec![512, 256, 128, 64];
    let k = 5;
    
    let model = ConvGCNUNet::new(
        in_dim, &gcn_dims, k, embedding_dim, true, "l2".to_string(), vb
    )?;

    // NOTE: In production, you'd load mu_x, std_x, mu_y, std_y from a safetensors config.
    // For now, we mock them.
    let scales: Vec<f32> = (1..104).map(|x| x as f32).collect(); // Example scales
    let mu_x = Tensor::zeros(103, DType::F32, &device)?;
    let std_x = Tensor::ones(103, DType::F32, &device)?;

    let regressor = UNetRegressor::new(
        model, 
        scales, 
        RealWavelet::Shannon, 
        mu_x, std_x, 
        0.0, 1.0, // mu_y, std_y
        1024,     // size
        512,      // stride
        device
    );

    // ---------------------------------------------------------
    // 4. Run Inference
    // ---------------------------------------------------------
    println!("Running inference (Continuous Wavelets + UNet)...");
    let distance_matrices = regressor.predict(&x_ndarray, &pos_ndarray, None)?;
    
    // distance_matrices shape: (Sequence_Length, NumPairs)
    println!("Generated coalescent distances for {} windows.", distance_matrices.nrows());

    // ---------------------------------------------------------
    // 5. Build Tree Sequence (Placeholder)
    // ---------------------------------------------------------
    println!("Writing Tree Sequence to {:?}", args.output);
    // TODO: We need to port your `distmat_to_tree` logic here.
    // The regression gives us pairwise coalescent times. We must apply a clustering 
    // algorithm (UPGMA / NJ) to build the trees and record them into a `tskit` TableCollection.

    Ok(())
}