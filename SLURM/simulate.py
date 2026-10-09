# -*- coding: utf-8 -*-
import os
import sys
import argparse
import logging
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="SLURM submitter for LMDB simulations")
    
    # Environment arguments
    parser.add_argument("--python_exec", default=sys.executable, 
                        help="Path to the Python executable (defaults to current env).")
    parser.add_argument("--simulate_script", default="python/simulate.py", 
                        help="Path to the simulator script.")
    
    # SLURM arguments
    parser.add_argument("--n_jobs", default=4, type=int, 
                        help="Total number of jobs to submit to SLURM.")
    parser.add_argument("--mem", default="32G", type=str,
                        help="Memory limit per job (e.g., 32G, 64G).")
    parser.add_argument("--time", default="2-00:00:00", type=str,
                        help="Time limit per job (SLURM format: days-hh:mm:ss, default 2 days).")
    
    # Passthrough arguments for simulate.py
    parser.add_argument("--n_replicates", default=10, type=int, 
                        help="Number of replicates PER JOB.")
    parser.add_argument("--ifile", default="None", 
                        help="Input config file for simulator.")
    parser.add_argument("--odir", default="sim_output", 
                        help="Output directory for LMDB database.")
    
    parser.add_argument("--discoal", action="store_true", 
                        help="Use Discoal simulator (passed to simulate.py)")
    parser.add_argument("--write_vcf", action="store_true", 
                        help="Write VCF files (passed to simulate.py)")
    parser.add_argument("--verbose", action="store_true", 
                        help="Display verbose messages in the runner")

    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
    return args

def main():
    args = parse_args()
    
    # 1. Validate the script exists
    if not os.path.exists(args.simulate_script):
        logging.error(f"Simulation script not found at: {args.simulate_script}")
        sys.exit(1)
        
    # 2. Ensure output directory exists ahead of time 
    if args.odir != "None":
        os.makedirs(args.odir, exist_ok=True)
        logging.debug(f"Runner ensured output directory exists: {args.odir}")

    logging.info(f"Submitting {args.n_jobs} jobs to SLURM.")
    logging.info(f"Resources per job: --mem={args.mem} -t {args.time}")
    
    # 3. Submit jobs to SLURM
    for job_id in range(args.n_jobs):
        # Build the python command string
        python_cmd_parts = [
            args.python_exec,
            args.simulate_script,
            "--job_id", str(job_id),
            "--n_replicates", str(args.n_replicates),
            "--ifile", args.ifile,
            "--odir", args.odir
        ]
        
        if args.discoal:
            python_cmd_parts.append("--discoal")
        if args.write_vcf:
            python_cmd_parts.append("--write_vcf")
            
        python_cmd = " ".join(python_cmd_parts)
        
        # Build the sbatch command
        sbatch_cmd = [
            "sbatch",
            f"--mem={args.mem}",
            "-t", args.time,
            "--wrap", python_cmd
        ]
        
        logging.debug(f"Executing: {' '.join(sbatch_cmd)}")
        
        try:
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
            # sbatch typically outputs "Submitted batch job 123456"
            logging.info(f"Job {job_id} submitted: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to submit Job {job_id}. Exit code {e.returncode}.\nStderr: {e.stderr}")

if __name__ == '__main__':
    main()