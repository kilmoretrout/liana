# -*- coding: utf-8 -*-
import argparse
import os
import csv
import json
import logging
import gc
from pathlib import Path
import numpy as np
import torch
from cyvcf2 import VCF
import pickle

# Your liana modeling modules
from liana.models.layers import ConvGCNUNet
from liana.models.regressors import UNetRegressor
from popgenml.data.functions import flip
from liana import liana

def parse_args():
    parser = argparse.ArgumentParser(description="Apply Liana to real VCF data and export to TSV.")
    
    # I/O Arguments
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the input .vcf.gz file.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to save the contiguous TSV file (e.g., output.tsv).")
    parser.add_argument("--verbose", action="store_true", help="Print inference progress.")
    parser.add_argument("--haploid", action="store_true", help="Use if the VCF contains haploid data.")
    
    # Subsetting Argument
    parser.add_argument("--n", type=int, default=None, 
                        help="Number of diploid pairs to evaluate on. Subsets to the first 2*n haploids.")
    
    # Custom Model Arguments 
    parser.add_argument("-w", "--weights", default="models/liana_dros_masked/best_model.pth", 
                        help="Path to PyTorch weights")
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
        
    return args

def extract_haplotypes(vcf_iterator, haploid=False):
    """
    Extracts positional data and the haplotype matrix from a cyvcf2 region iterator.
    Filters out non-biallelic SNPs. Maps missing genotypes (-1) to 0.5.
    """
    x_list = []
    pos_list = []
    
    for variant in vcf_iterator:
        if not variant.is_snp or len(variant.ALT) > 1:
            continue
            
        gts = variant.genotypes
        haplotypes = []
        for gt in gts:
            if haploid:
                val = 0.5 if gt[0] == -1 else gt[0]
                haplotypes.append(val)
            else:
                val1 = 0.5 if gt[0] == -1 else gt[0]
                val2 = 0.5 if gt[1] == -1 else gt[1]
                haplotypes.append(val1)
                haplotypes.append(val2)
                
        x_list.append(haplotypes)
        pos_list.append(variant.POS)
        
    if not x_list:
        return None, None
        
    x = np.array(x_list, dtype=np.float32).T 
    pos = np.array(pos_list, dtype=np.float32)
    return x, pos

def main():
    args = parse_args()
    
    # ---------------------------------------------------------
    # Load Model Hyperparameters from JSON
    # ---------------------------------------------------------
    weights_path = Path(args.weights)
    json_path = weights_path.parent / "arguments.json"
    
    if json_path.exists():
        with open(json_path, "r") as f:
            config = json.load(f)
            parsed_args = config.get("parsed_args", {})
            
            for key, value in parsed_args.items():
                if not (key in ["weights", "input", "output", "n"]):
                    setattr(args, key, value)
                
        if hasattr(args, 'L'):
            args.window_size = args.L
            
        logging.info(f"Successfully loaded model hyper-parameters from {json_path}")
    else:
        raise FileNotFoundError(f"arguments.json not found at {json_path}! Cannot determine window size (L).")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using {device} for liana model inference")

    # ---------------------------------------------------------
    # Open VCF and Parse Contig Lengths
    # ---------------------------------------------------------
    vcf = VCF(args.input)
    
    contigs = []
    for header_line in vcf.header_iter():
        if header_line['HeaderType'] == 'CONTIG':
            contig_id = header_line['ID']
            length = int(header_line['length'])
            contigs.append((contig_id, length))
            
    if not contigs:
        raise ValueError("No contig lengths found in the VCF header.")
        
    # Assume we are operating on the primary contig in the VCF
    primary_contig = contigs[0][0]
    chrom_length = float(contigs[0][1])
    logging.info(f"Processing entire VCF for contig {primary_contig} (Length: {chrom_length})")

    # ---------------------------------------------------------
    # Extract & Filter Haplotypes
    # ---------------------------------------------------------
    # By passing the base VCF object, it acts as an iterator over the entire file
    x, pos = extract_haplotypes(vcf, haploid=args.haploid)
    
    if x is None or x.shape[1] == 0:
        logging.error("No valid biallelic SNPs found in the provided VCF.")
        return
        
    # Subset to first 2*n haploids if requested
    if args.n is not None:
        target_haps = 2 * args.n
        if target_haps < x.shape[0]:
            x = x[:target_haps, :]
            
    # Requires at least one 0 and at least one 1 (ignores 0.5 missing calls)
    is_ref = (x == 0)
    is_alt = (x == 1)
    valid_sites = np.where(is_ref.any(axis=0) & is_alt.any(axis=0))[0]
    
    if len(valid_sites) < 2:
        logging.error("Not enough valid polymorphic sites after filtering.")
        return
        
    x = x[:, valid_sites]
    pos = pos[valid_sites]
    x = flip(x)
    
    # ---------------------------------------------------------
    # Run Global Inference & Write to TSV
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with open(args.output, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(['num_snps', 'newick'])
        
        try:
            # Pass actual chromosome length for L, and the JSON args.L for window_size
            logging.info("Starting liana inference...")
            ts_est_liana = liana(args.weights, x, pos, chrom_length, float(args.window_size), device_str=str(device), verbose = True)
            
            # Extract the PGTreeSequence object
            pg_ts = getattr(ts_est_liana, 'ts', ts_est_liana)
            
            logging.info("Writing trees to TSV...")
            # Iterate through the compressed bytes and matched SNP counts
            for t_bytes, num_snps in zip(pg_ts._tree_bytes, pg_ts.snps):
                ts_reloaded = pickle.loads(t_bytes)
                tree = ts_reloaded.first()
                writer.writerow([num_snps, tree.newick()])
                
            logging.info(f"Successfully saved to {args.output}")
                
        except Exception as e:
            logging.error(f"Inference failed: {e}")

if __name__ == '__main__':
    main()