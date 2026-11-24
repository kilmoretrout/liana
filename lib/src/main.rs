use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
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
    pub clades: Vec<String>, // Tracks the canonical string for each internal node
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
    
    let mut clades = Vec::with_capacity(n - 1);
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

        let mut children = [topologies[min_i].clone(), topologies[min_j].clone()];
        children.sort(); 
        topologies[next_node] = format!("({},{})", children[0], children[1]);
        
        // Save this node's canonical topology as a distinct clade
        clades.push(topologies[next_node].clone());

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
        clades,
        merges,
        node_times,
    }
}

// ---------------------------------------------------------
// Clade Smoothing & Harmonic Averaging
// ---------------------------------------------------------
fn average_clade_times(trees: &[Tree]) -> Vec<Vec<f32>> {
    let mut runs: Vec<(Vec<f32>, Vec<usize>)> = Vec::new();
    let mut active_runs: HashMap<&str, usize> = HashMap::new(); // Zero-allocation tracking!
    
    // Pass 1: Extract clades and group into contiguous runs
    for (tree_idx, tree) in trees.iter().enumerate() {
        let mut current_clades = HashMap::new();
        for (i, clade) in tree.clades.iter().enumerate() {
            current_clades.insert(clade.as_str(), tree.node_times[i]);
        }
        
        let current_clade_set: HashSet<&str> = current_clades.keys().copied().collect();
        let active_clade_set: HashSet<&str> = active_runs.keys().copied().collect();
        
        // End tracking for clades no longer in current tree
        for clade in active_clade_set.difference(&current_clade_set) {
            active_runs.remove(clade);
        }
        
        // Update continuing and start new runs
        for (clade, time) in current_clades {
            let run_idx = *active_runs.entry(clade).or_insert_with(|| {
                let idx = runs.len();
                runs.push((Vec::new(), Vec::new()));
                idx
            });
            runs[run_idx].0.push(time);
            runs[run_idx].1.push(tree_idx);
        }
    }
    
    // Pass 2: Calculate harmonic averages
    let mut result = vec![Vec::with_capacity(trees[0].clades.len()); trees.len()];
    
    for (times, tree_indices) in runs {
        let mut has_zero = false;
        let mut sum_inv = 0.0;
        
        for &t in &times {
            if t <= 0.0 {
                has_zero = true;
                break;
            }
            sum_inv += 1.0 / t;
        }
        
        let avg_time = if has_zero || sum_inv == 0.0 {
            0.0
        } else {
            (times.len() as f32) / sum_inv
        };
        
        for &idx in &tree_indices {
            result[idx].push(avg_time);
        }
    }
    
    // Sort times for each tree to ensure monotonic increasing heights for UPGMA
    for res in &mut result {
        res.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    
    result
}

// Reconstructs a standard Newick string using the SMOOTHED coalescent times
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
        
        let mut first_val = -1.0;
        let mut is_uniform = true;
        
        for i in 9..(9 + n) {
            let gt_base = cols[i].split(':').next().unwrap_or("0");
            let val = if gt_base.contains('.') { 0.5 } else if gt_base.contains('1') { 1.0 } else { 0.0 };
            
            // Skip missing data when determining if the site is uniform
            if val == 0.5 {
                continue; 
            }
            
            // Lock in the first non-missing value we see
            if first_val == -1.0 {
                first_val = val;
            } else if val != first_val {
                // We found a non-missing value that differs from our locked value
                is_uniform = false;
                break;
            }
        }
        
        // If the site is NOT uniform (i.e. it's polymorphic), save the position
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

    // 4. Clade Smoothing (Bidirectional Harmonic Averaging)
    println!("Smoothing coalescent times bidirectionally...");
    let forward_times = average_clade_times(&trees);
    
    let mut rev_trees = trees.clone();
    rev_trees.reverse();
    let mut backward_times = average_clade_times(&rev_trees);
    backward_times.reverse(); // Flip back to match original orientation

    let mut smoothed_times = vec![vec![0.0; n - 1]; sequence_length];
    for i in 0..sequence_length {
        for j in 0..(n - 1) {
            smoothed_times[i][j] = (forward_times[i][j] + backward_times[i][j]) / 2.0;
        }
    }

    // 5. Collapse Regions and Write TSV
    println!("Collapsing identical topologies and writing TSV...");
    let out_file = File::create(&args.output)?;
    let mut writer = BufWriter::new(out_file);
    writeln!(writer, "Start_BP\tEnd_BP\tNewick")?;

    let mut current_topology = trees[0].topology_hash.clone();
    let mut current_merges = trees[0].merges.clone();
    let mut start_idx = 0;
    let mut num_regions = 0;

    // Because times are perfectly smoothed across identical topologies, 
    // we can just pick the times at `start_idx` to build the Newick tree!
    for i in 1..sequence_length {
        if trees[i].topology_hash != current_topology {
            let newick = build_newick(n, &current_merges, &smoothed_times[start_idx]);
            writeln!(writer, "{:.0}\t{:.0}\t{}", pos_vec[start_idx], pos_vec[i], newick)?;
            num_regions += 1;

            current_topology = trees[i].topology_hash.clone();
            current_merges = trees[i].merges.clone();
            start_idx = i;
        }
    }

    // Write final region
    let newick = build_newick(n, &current_merges, &smoothed_times[start_idx]);
    writeln!(writer, "{:.0}\t{:.0}\t{}", pos_vec[start_idx], pos_vec[sequence_length - 1], newick)?;
    num_regions += 1;
    
    writer.flush()?;
    println!("Done! Generated contiguous tree sequence with {} regions.", num_regions);

    Ok(())
}