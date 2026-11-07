# -*- coding: utf-8 -*-
"""
SLURM sbatch submission script for benchmarking LMDB simulations.

Discovers all .lmdb directories/files in --lmdb_dir, generates a dedicated SLURM
batch script for each database (requesting 24G RAM and 2-00:00:00 walltime), and
submits jobs with unique indexed CSV outputs in --odir.
"""

import argparse
import glob
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit sbatch array/jobs for each .lmdb in a directory to benchmark."
    )

    # Submission & I/O
    parser.add_argument(
        "--benchmark_script",
        type=str,
        default="src/liana/cli/benchmark.py",
        help="Path to the python benchmark script.",
    )
    parser.add_argument(
        "--lmdb_dir",
        type=str,
        required=True,
        help="Directory containing .lmdb files or directories.",
    )
    parser.add_argument(
        "--odir",
        type=str,
        required=True,
        help="Output directory where indexed results CSVs and SLURM logs will be saved.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Generate batch scripts without invoking sbatch.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="Optional SLURM partition/queue name.",
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=1,
        help="CPUs per task requested from SLURM. Default: 1.",
    )

    # Benchmark Script Flags & Parameters
    parser.add_argument("--verbose", action="store_true", help="Print inference progress.")
    parser.add_argument("--haploid", action="store_true", help="Use if haploid data.")
    parser.add_argument("--thin", default=20, type=int, help="Thinning interval for MCMC.")
    parser.add_argument(
        "-b",
        "--bench_against",
        choices=["all", "singer", "relate"],
        default="all",
        help="Which baseline tools to benchmark against Liana. Default: 'all'.",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=None,
        help="Number of replicates/chunks to process per database.",
    )
    parser.add_argument(
        "--n",
        default=2,
        type=int,
        help="Number of diploid pairs to evaluate on.",
    )
    parser.add_argument(
        "--tsq_builder",
        default="./target/release/tsq_builder",
        help="Path to compiled Rust tsq_builder executable.",
    )
    parser.add_argument(
        "--mu",
        default=1.5e-8,
        type=float,
        help="Default mutation rate.",
    )
    parser.add_argument(
        "--r",
        default=1.007e-8,
        type=float,
        help="Default recombination rate.",
    )
    parser.add_argument(
        "-w",
        "--weights",
        default="models/n16_k11_cheby/best_model.pth",
        help="Path to PyTorch weights.",
    )
    parser.add_argument(
        "--two_pop",
        action="store_true",
        help="For two-pop data.",
    )
    parser.add_argument(
        "--skip_liana",
        action="store_true",
        help="Skip running the Liana model entirely.",
    )
    parser.add_argument(
        "--L",
        default=2.5e6,
        type=float,
        help="Length in bp of simulated chroms.",
    )

    return parser.parse_args()


def build_forwarded_args(args) -> list:
    """Builds the list of CLI arguments to forward to each benchmark run."""
    forwarded = []

    if args.verbose:
        forwarded.append("--verbose")
    if args.haploid:
        forwarded.append("--haploid")
    if args.two_pop:
        forwarded.append("--two_pop")
    if args.skip_liana:
        forwarded.append("--skip_liana")

    forwarded.extend(["--thin", str(args.thin)])
    forwarded.extend(["-b", args.bench_against])
    if args.n_samples is not None:
        forwarded.extend(["-n", str(args.n_samples)])
    forwarded.extend(["--n", str(args.n)])

    forwarded.extend(["--tsq_builder", args.tsq_builder])
    forwarded.extend(["--mu", str(args.mu)])
    forwarded.extend(["--r", str(args.r)])
    forwarded.extend(["-w", args.weights])
    forwarded.extend(["--L", str(args.L)])

    return forwarded


def main():
    args = parse_args()

    odir = os.path.abspath(args.odir)
    os.makedirs(odir, exist_ok=True)

    slurm_scripts_dir = os.path.join(odir, "slurm_scripts")
    slurm_logs_dir = os.path.join(odir, "slurm_logs")
    os.makedirs(slurm_scripts_dir, exist_ok=True)
    os.makedirs(slurm_logs_dir, exist_ok=True)

    # Discover LMDB files / directories
    lmdb_pattern = os.path.join(args.lmdb_dir, "*.lmdb")
    lmdb_paths = sorted(glob.glob(lmdb_pattern))

    if not lmdb_paths:
        print(f"Error: No .lmdb targets found matching '{lmdb_pattern}'", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(lmdb_paths)} LMDB targets in {args.lmdb_dir}")
    benchmark_script_path = os.path.abspath(args.benchmark_script)
    forwarded_flags = build_forwarded_args(args)

    for idx, lmdb_path in enumerate(lmdb_paths):
        base_name = os.path.splitext(os.path.basename(lmdb_path))[0]
        output_csv = os.path.join(odir, f"{idx:04d}_{base_name}_results.csv")
        job_name = f"bench_{idx:04d}_{base_name[:15]}"
        script_path = os.path.join(slurm_scripts_dir, f"submit_{idx:04d}_{base_name}.sh")
        log_out = os.path.join(slurm_logs_dir, f"{job_name}_%j.out")
        log_err = os.path.join(slurm_logs_dir, f"{job_name}_%j.err")

        cmd_parts = [
            sys.executable,
            benchmark_script_path,
            "-i", os.path.abspath(lmdb_path),
            "-o", output_csv,
        ] + forwarded_flags

        cmd_line = " ".join(cmd_parts)

        # Build sbatch header
        sbatch_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={log_out}",
            f"#SBATCH --error={log_err}",
            f"#SBATCH --time=2-00:00:00",
            f"#SBATCH --mem=24G",
            f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        ]
        if args.partition:
            sbatch_lines.append(f"#SBATCH --partition={args.partition}")

        sbatch_lines.extend([
            "",
            "echo \"Starting benchmark job ${SLURM_JOB_ID} on $(hostname) at $(date)\"",
            f"echo \"Input LMDB: {lmdb_path}\"",
            f"echo \"Output CSV: {output_csv}\"",
            "",
            cmd_line,
            "",
            "EXIT_CODE=$?",
            "echo \"Finished benchmark job with exit code ${EXIT_CODE} at $(date)\"",
            "exit ${EXIT_CODE}",
            "",
        ])

        with open(script_path, "w") as f:
            f.write("\n".join(sbatch_lines))
        os.chmod(script_path, 0o755)

        if args.dry_run:
            print(f"[DRY-RUN] Created: {script_path} -> {output_csv}")
        else:
            try:
                res = subprocess.run(
                    ["sbatch", script_path],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(f"Submitted [{idx + 1}/{len(lmdb_paths)}]: {res.stdout.strip()} ({script_path})")
            except subprocess.CalledProcessError as e:
                print(f"Failed to submit {script_path}: {e.stderr.strip()}", file=sys.stderr)


if __name__ == "__main__":
    main()
