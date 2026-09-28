use clap::Parser;
use std::path::PathBuf;
use std::fs::File;
use std::io::{BufReader, Write, BufRead};

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


// (Assuming your other imports are here)

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
    
    let file = File::open(&args.input)?;
    let mut reader = noodles::bgzf::Reader::new(file);
    
    let mut line = String::new();
    let mut n_individuals = 0;
    let mut total_base_pairs = 0.0f32;
    
    // 1. Read header to get the number of individuals AND contig length
    while reader.read_line(&mut line)? > 0 {
        // Try to grab the exact chromosome length if the VCF header provides it
        if line.starts_with("##contig=") && line.contains("length=") {
            if let Some(start_idx) = line.find("length=") {
                let tail = &line[start_idx + 7..];
                let end_idx = tail.find(['>', ',']).unwrap_or(tail.len());
                if let Ok(l) = tail[..end_idx].parse::<f32>() {
                    total_base_pairs = total_base_pairs.max(l);
                }
            }
        } else if line.starts_with("#CHROM") {
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
            let gt_str = cols[i].split(':').next().unwrap_or("0");
            
            // Check for missing data first
            let allele = if gt_str.contains('.') {
                0.5f32
            } else if gt_str.contains('1') { 
                1.0f32 
            } else { 
                0.0f32 
            };
            genotype_matrix.push(allele);
        }
        
        line.clear();
    }

    let sequence_length = pos_vec.len();
    
    // Fallback: If the VCF header lacked the contig length, use the position of the last SNP.
    if let Some(&last_snp_pos) = pos_vec.last() {
        total_base_pairs = total_base_pairs.max(last_snp_pos);
    }
    
    println!(
        "Parsed {} variants spanning {:.2} Megabases for {} individuals.", 
        sequence_length, 
        total_base_pairs / 1_000_000.0, 
        n_individuals
    );

    // Reshape the flat vector into an (N, L) matrix
    let x_ndarray = Array2::from_shape_vec((n_individuals, sequence_length), genotype_matrix)?;
    let pos_ndarray = Array1::from_vec(pos_vec);

    // ---------------------------------------------------------
    // 2.5 Downsample & Filter Uniform Sites
    // ---------------------------------------------------------
    // ---------------------------------------------------------
    // 2.5 Downsample & Filter Uniform Sites
    // ---------------------------------------------------------
    println!("Downsampling to 32 individuals and removing uniform sites...");
    use ndarray::s;
    
    let target_individuals = 32.min(n_individuals);
    let mut filtered_x = Vec::new();
    let mut filtered_pos = Vec::new();

    for j in 0..sequence_length {
        // Look at the j-th SNP for only the first 32 individuals
        let column = x_ndarray.slice(s![0..target_individuals, j]);
        
        // A site is uniform if every individual has the exact same value.
        let first_val = column[0];
        let is_uniform = column.iter().all(|&val| val == first_val);
        
        // Keep the site only if it has variation
        if !is_uniform {
            filtered_pos.push(pos_ndarray[j]);
            filtered_x.extend(column.iter());
        }
    }

    let new_sequence_length = filtered_pos.len();
    
    // Replace the old arrays with the new downsampled ones
    let x_ndarray = Array2::from_shape_vec((target_individuals, new_sequence_length), filtered_x)?;
    let pos_ndarray = Array1::from_vec(filtered_pos);
    
    // Update n_individuals and sequence_length variables for the rest of the script
    let n_individuals = target_individuals;
    let sequence_length = new_sequence_length;

    println!(
        "After filtering, kept {} polymorphic variants across {} individuals.", 
        sequence_length, n_individuals
    );
    
    // Update n_individuals and sequence_length variables for the rest of the script
    let n_individuals = target_individuals;
    let sequence_length = new_sequence_length;

    println!(
        "After filtering, kept {} polymorphic variants across {} individuals.", 
        sequence_length, n_individuals
    );

    // ---------------------------------------------------------
    // 3. Load Neural Network Weights
    // ---------------------------------------------------------
    println!("Loading model weights from {:?}", args.model_weights);
    let vb = unsafe { 
        VarBuilder::from_mmaped_safetensors(&[&args.model_weights], DType::F32, &device)? 
    };

    let in_dim = 101;
    let embedding_dim = 32;
    let gcn_dims = vec![512, 256, 128, 64];
    let k = 11;
    
    let model = ConvGCNUNet::new(
        in_dim, &gcn_dims, k, embedding_dim, false, "lorentz".to_string(), vb
    )?;

    // NOTE: In production, you'd load mu_x, std_x, mu_y, std_y from a safetensors config.
    let scales: Vec<f32> = (1..100).map(|x| x as f32).collect(); 
    let mu_x = Tensor::zeros(99, DType::F32, &device)?;
    let std_x = Tensor::ones(99, DType::F32, &device)?;

    let regressor = UNetRegressor::new(
        model, 
        scales, 
        RealWavelet::Shannon, 
        mu_x, std_x, 
        0.0, 1.0, 
        1024,     
        512,      
        device
    );

    // ---------------------------------------------------------
    // 4. Run Inference
    // ---------------------------------------------------------
    println!("Running inference (Continuous Wavelets + UNet)...");
    

    let temp_bin_path = "predictions_output.bin";
    
    regressor.predict_to_disk(
        &x_ndarray, 
        &pos_ndarray, 
        None, 
        total_base_pairs,  // <--- Pass the parsed variable here!
        temp_bin_path
    )?;
    
    println!("Successfully generated and streamed coalescent distances to: {}", temp_bin_path);

    // ---------------------------------------------------------
    // 5. Build Tree Sequence
    // ---------------------------------------------------------
    println!("Writing Tree Sequence to {:?}", args.output);
    // TODO: Memory-map `temp_bin_path` and run UPGMA/NJ clustering here.

    Ok(())
}