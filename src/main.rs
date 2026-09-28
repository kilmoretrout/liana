use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufWriter, Write};
use memmap2::MmapOptions;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    vcf: String,
    
    #[arg(short, long)]
    bin: String,
    
    #[arg(short, long, default_value = "tree_sequence_regions.tsv")]
    output: String,

    #[arg(short, long, default_value_t = 32)]
    target_ind: usize,
}

#[inline]
fn get_idx(n: usize, i: usize, j: usize) -> usize {
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    (n * a) - (a * (a + 1) / 2) + b - a - 1
}

#[derive(Clone, Debug)]
pub struct Tree {
    pub topology_hash: String,
    pub merges: Vec<(usize, usize)>,
    pub node_times: Vec<f32>,
}

pub fn run_upgma(condensed_dist: &[f32], n: usize) -> Tree {
    let mut dist = vec![vec![0.0; n * 2]; n * 2];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = condensed_dist[get_idx(n, i, j)];
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    let mut active = vec![true; n * 2];
    let mut cluster_size = vec![1.0; n * 2];
    let mut topologies = vec![String::new(); n * 2];
    
    let mut merges = Vec::with_capacity(n - 1);
    let mut node_times = Vec::with_capacity(n - 1);
    
    for i in 0..n {
        topologies[i] = i.to_string();
    }

    let mut next_node = n;

    for _ in 0..(n - 1) {
        let mut min_d = f32::MAX;
        let mut min_i = 0;
        let mut min_j = 0;

        for i in 0..next_node {
            if !active[i] { continue; }
            for j in (i + 1)..next_node {
                if !active[j] { continue; }
                if dist[i][j] < min_d {
                    min_d = dist[i][j];
                    min_i = i;
                    min_j = j;
                }
            }
        }

        active[min_i] = false;
        active[min_j] = false;
        active[next_node] = true;

        node_times.push(min_d / 2.0);

        // Sorting the string representations creates a canonical topology hash
        // guaranteeing that ((0,1),2) and (2,(1,0)) map to the exact same string
        let mut children = [topologies[min_i].clone(), topologies[min_j].clone()];
        children.sort(); 
        topologies[next_node] = format!("({},{})", children[0], children[1]);

        // Keep merge tuples sorted identically for Newick generation later
        if topologies[min_i] < topologies[min_j] {
            merges.push((min_i, min_j));
        } else {
            merges.push((min_j, min_i));
        }

        let size_i = cluster_size[min_i];
        let size_j = cluster_size[min_j];
        cluster_size[next_node] = size_i + size_j;

        for k in 0..next_node {
            if active[k] {
                let d = (size_i * dist[min_i][k] + size_j * dist[min_j][k]) / (size_i + size_j);
                dist[next_node][k] = d;
                dist[k][next_node] = d;
            }
        }
        next_node += 1;
    }

    Tree {
        topology_hash: topologies[next_node - 1].clone(),
        merges,
        node_times,
    }
}

// Reconstructs a standard Newick string using the AVERAGED coalescent times
fn build_newick(n: usize, merges: &[(usize, usize)], node_times: &[f32]) -> String {
    let mut heights = vec![0.0; n * 2];
    let mut strings = vec![String::new(); n * 2];
    
    for i in 0..n {
        strings[i] = i.to_string();
    }
    
    for i in 0..(n - 1) {
        let parent = n + i;
        heights[parent] = node_times[i];
        
        let (c1, c2) = merges[i];
        
        let bl1 = (heights[parent] - heights[c1]).max(0.0);
        let bl2 = (heights[parent] - heights[c2]).max(0.0);
        
        strings[parent] = format!("({}:{},{}:{})", strings[c1], bl1, strings[c2], bl2);
    }
    
    format!("{};", strings[(n * 2) - 2])
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let n = args.target_ind;

    // 1. Extract exactly the positions that the model predicted on
    println!("Extracting positions from VCF (filtering to {} individuals)...", n);
    let vcf_file = File::open(&args.vcf)?;
    let mut reader = noodles::bgzf::Reader::new(vcf_file);
    let mut line = String::new();
    
    let mut pos_vec = Vec::new();

    while reader.read_line(&mut line)? > 0 {
        if line.starts_with('#') { line.clear(); continue; }
        
        let cols: Vec<&str> = line.trim_end().split('\t').collect();
        if cols.len() < 9 + n { line.clear(); continue; }
        
        let pos: f32 = cols[1].parse()?;
        
        // Emulate Python's `if not np.all(row_arr == row_arr[0]):`
        let mut first_val = -1.0;
        let mut is_uniform = true;
        
        for i in 9..(9 + n) {
            let gt_base = cols[i].split(':').next().unwrap_or("0");
            let val = if gt_base.contains('.') { 0.5 } else if gt_base.contains('1') { 1.0 } else { 0.0 };
            
            if i == 9 {
                first_val = val;
            } else if val != first_val {
                is_uniform = false;
                break; // Short circuit as soon as variation is found
            }
        }
        
        if !is_uniform {
            pos_vec.push(pos);
        }
        line.clear();
    }
    
    let sequence_length = pos_vec.len();
    let num_pairs = (n * (n - 1)) / 2;
    println!("Found {} polymorphic sites.", sequence_length);

    // 2. Memory-map the Binary file
    println!("Memory-mapping {}...", args.bin);
    let bin_file = File::open(&args.bin)?;
    let mmap = unsafe { MmapOptions::new().map(&bin_file)? };
    
    let f32_slice = unsafe {
        std::slice::from_raw_parts(
            mmap.as_ptr() as *const f32, 
            mmap.len() / std::mem::size_of::<f32>()
        )
    };
    
    assert_eq!(f32_slice.len(), sequence_length * num_pairs, "Mismatch between VCF and BIN lengths!");

    // 3. Parallel UPGMA Construction
    println!("Spawning UPGMA tasks across all CPU threads...");
    let trees: Vec<Tree> = (0..sequence_length)
        .into_par_iter()
        .map(|w| {
            let dists = &f32_slice[w * num_pairs .. (w + 1) * num_pairs];
            run_upgma(dists, n)
        })
        .collect();

    // 4. Collapse Regions and Write TSV
    println!("Collapsing identical topologies and writing TSV...");
    let out_file = File::create(&args.output)?;
    let mut writer = BufWriter::new(out_file);
    writeln!(writer, "Start_BP\tEnd_BP\tNewick")?;

    let mut current_topology = trees[0].topology_hash.clone();
    let mut current_merges = trees[0].merges.clone();
    let mut current_times = trees[0].node_times.clone();
    let mut start_idx = 0;
    let mut count = 1.0;

    let mut num_regions = 0;

    for i in 1..sequence_length {
        if trees[i].topology_hash == current_topology {
            for (acc, t) in current_times.iter_mut().zip(&trees[i].node_times) {
                *acc += t;
            }
            count += 1.0;
        } else {
            // Region boundary! Average times and write Newick
            for acc in current_times.iter_mut() { *acc /= count; }
            
            let newick = build_newick(n, &current_merges, &current_times);
            writeln!(writer, "{:.0}\t{:.0}\t{}", pos_vec[start_idx], pos_vec[i], newick)?;
            num_regions += 1;

            // Reset
            current_topology = trees[i].topology_hash.clone();
            current_merges = trees[i].merges.clone();
            current_times = trees[i].node_times.clone();
            start_idx = i;
            count = 1.0;
        }
    }

    // Write final region
    for acc in current_times.iter_mut() { *acc /= count; }
    let newick = build_newick(n, &current_merges, &current_times);
    writeln!(writer, "{:.0}\t{:.0}\t{}", pos_vec[start_idx], pos_vec[sequence_length - 1], newick)?;
    num_regions += 1;
    
    writer.flush()?;
    println!("Done! Generated contiguous tree sequence with {} regions.", num_regions);

    Ok(())
}