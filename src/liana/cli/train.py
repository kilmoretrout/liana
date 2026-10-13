# -*- coding: utf-8 -*-
import os
import argparse
import logging
import itertools
import numpy as np
import time
import random

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning:msprime"

import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from itertools import islice
import pandas as pd

# Custom Modules
from popgenml.data.functions import flip
import pgml
from layers import ConvGCNUNet

import warnings
# Suppress msprime spam
logging.getLogger("msprime").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*msprime.TreeSequence.*", category=FutureWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--idir", default="None")
    
    parser.add_argument("--min_log", default=1, type=int)
    parser.add_argument("--max_log", default=10, type=int)
    parser.add_argument("--n_scales", default=128, type=int)
    parser.add_argument("--wavelet", default="shannon")
    parser.add_argument("--wav_param", default=5.0, type=float) # Added so it doesn't crash if passed
    
    parser.add_argument("--size", default=256, type=int)
    parser.add_argument("--n_replicates", default=-1, type=int)
    parser.add_argument("--n_samples", default=4, type=int)    
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--n_batch", default=3, type=int)
    parser.add_argument("--dilation", default=0., type = float)
    
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension for ConvGCNUNet")
    parser.add_argument("--k", type=int, default=16, help="K value for the graph / kNN")
    
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of training epochs")
    parser.add_argument("--two_pop", action = "store_true")
    parser.add_argument("--weights", default="None")
    parser.add_argument("--model", default = "conv")
    parser.add_argument("--n", default = 2, type = int)
    parser.add_argument("--factor", default = 2.0, type = float)
    parser.add_argument("--L", default = 2.5e6, type = float)
    
    parser.add_argument("--n_workers", default = 8, type = int)
    parser.add_argument("--prefetch", default = 4, type = int)
    parser.add_argument("--recompute_means", action = "store_true")
    parser.add_argument("--masks", default = "None")
    parser.add_argument("--rec", action = "store_true")
                            
    parser.add_argument("--dist", default="l2")
    parser.add_argument("--odir", default="None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.makedirs(args.odir, exist_ok=True)
            logging.debug(f'root: made output directory {args.odir}')
            
    return args

def worker_init_fn(worker_id):
    """Ensures background workers get unique seeds across all GPUs."""
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    worker_seed = (torch.initial_seed() + global_rank) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==========================================
# DATASET & STATS
# ==========================================
import os
import glob
import torch

def get_condensed_indices(indices, n):
    """
    Calculates the 1D indices in a condensed distance matrix 
    for all pairwise combinations of the provided 'indices'.
    """
    # Sort indices to ensure i < j
    idx = sorted(indices)
    k_indices = []
    for i_map, i in enumerate(idx):
        for j in idx[i_map + 1:]:
            # The standard formula for condensed index
            k = n * i - i * (i + 1) // 2 + (j - i - 1)
            k_indices.append(k)
    return k_indices

class DiskWaveletDataset(IterableDataset):
    def __init__(self, args, file_list, L = 2.5e6, n = None, two_pop = False, dilation = 0., 
                 use_theory = False, mask_folder = None):
        super().__init__()
        self.args = args
        self.files = file_list # Now a list of .lmdb directory paths
        
        self.L = L
        self.dilation = dilation
            
        self.scales = np.unique((2 ** np.linspace(args.min_log, args.max_log, args.n_scales)).astype(np.int32))

        # Stats placeholders
        self.mean_x, self.std_x = None, None
        self.mean_y, self.std_y = None, None
        
        self.two_pop = two_pop
        
        self.n = n
        
        self.masks = []
        if mask_folder is not None and os.path.exists(mask_folder):
            mask_files = glob.glob(os.path.join(mask_folder, '*.npz'))
            for f in mask_files:
                try:
                    with np.load(f) as data:
                        # Grab the mask (defaulting to the first array in the npz if 'mask' isn't explicitly the key)
                        k = 'mask' if 'mask' in data else data.files[0]
                        # Store as boolean to save RAM
                        self.masks.append(data[k].astype(bool))
                except Exception as e:
                    print(f"Failed to load mask {f}: {e}")

    def inject_stats(self, mean_x, std_x, mean_y, std_y):
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y

    def __iter__(self):
        # Setup Wavelet Configs
        if self.args.wavelet == "shannon":
            config = pgml.WaveletConfig.shannon()
        elif self.args.wavelet == "gaussian":
            config = pgml.WaveletConfig.gaussian(order=int(self.args.wav_param))
        elif self.args.wavelet == "haar":
            config = pgml.WaveletConfig.haar()
        elif self.args.wavelet == "smc":
            config = pgml.WaveletConfig.smc()
        else:
            config = pgml.WaveletConfig.morlet(w0=self.args.wav_param)
            
        worker_info = torch.utils.data.get_worker_info()
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        
        if worker_info is None:
            worker_seed = int.from_bytes(os.urandom(4), byteorder="little") % (2**32 - 1)
            num_workers = 1
            worker_id = 0
        else:
            worker_seed = (torch.initial_seed() + global_rank) % (2**32 - 1)
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            
        np.random.seed(worker_seed)

        # 1. Gather all keys from all provided LMDB files
        # This is extremely fast (fractions of a second even for millions of keys)
        all_items = []
        for lmdb_path in self.files:
            try:
                env = lmdb.open(lmdb_path, readonly=True, lock=False)
                with env.begin() as txn:
                    # iternext(values=False) ensures we only load the small keys into memory
                    keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
                    all_items.extend([(lmdb_path, k) for k in keys])
                env.close()
            except Exception as e:
                print(f"Failed to read keys from {lmdb_path}: {e}")
                
        # 2. Shard the specific data points across workers
        # This prevents the bug where having fewer LMDB files than workers starves some workers
        worker_items = all_items[worker_id::num_workers]
        
        # 3. Shuffle the data points for random training distribution
        np.random.shuffle(worker_items)
        
        # 4. Open environments needed by this specific worker and keep them open
        envs = {}
        unique_paths = set(path for path, key in worker_items)
        for path in unique_paths:
            envs[path] = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)

        # 5. Iterate through the assigned keys
        for lmdb_path, key in worker_items:
            try:
                with envs[lmdb_path].begin() as txn:
                    byte_data = txn.get(key)
                
                data = pickle.loads(byte_data)
                
                x = data['x'].astype(np.float32)
                y = data['D']
                ints = data['intervals']
                pos = data['pos']
                    
            except Exception as e:
                print(f"Skipping key {key} in {lmdb_path} due to error: {e}")
                continue
            
            if self.n:
                if x.shape[0] != self.n * 2:
                    # randomly downsample the data (choosing some number of pairs)
                    total_pairs = x.shape[0] // 2
                    paired_indices = np.array([[2*u, 2*u + 1] for u in range(total_pairs)], dtype = np.int32)
                    
                    if getattr(self, 'two_pop', False): # Safely check if two_pop is True
                        # Split the target number of pairs in half
                        n1 = self.n // 2
                        n2 = self.n - n1
                        
                        mid_point = total_pairs // 2
                        
                        # Sample n1 pairs from the first half
                        idx1 = np.random.choice(mid_point, n1, replace=False)
                        # Sample n2 pairs from the second half
                        idx2 = np.random.choice(total_pairs - mid_point, n2, replace=False) + mid_point
                        
                        ii = np.concatenate([idx1, idx2])
                    else:
                        # Original logic: sample from the entire alignment
                        ii = np.random.choice(total_pairs, self.n, replace=False)
                        
                    ii = np.sort(paired_indices[ii].flatten())
                    
                    ii_ = get_condensed_indices(ii, x.shape[0])
                    
                    # downsample
                    x = x[ii]
                    # downsample the condensed distance matrix
                    y = y[:,ii_]
                                
                    # take out uniform sites
                    xs = x.sum(0)
                    ii = np.where((xs != 0) & (xs != x.shape[0]))[0]
                    
                    x = x[:,ii]
                    pos = pos[ii]
                    
            if self.masks and x.shape[1] > 0:
                L_x = x.shape[1]
                N_x = x.shape[0]
                
                # Pick a random mask
                chosen_mask = self.masks[np.random.randint(len(self.masks))]
                
                # Ensure the mask is long enough (tile columns if it's shorter than L_x)
                if chosen_mask.shape[1] < L_x:
                    repeats = (L_x // chosen_mask.shape[1]) + 1
                    chosen_mask = np.tile(chosen_mask, (1, repeats))
                
                # Randomly slice columns to match x.shape[1]
                max_start_col = chosen_mask.shape[1] - L_x
                start_col = np.random.randint(0, max_start_col + 1) if max_start_col > 0 else 0
                mask_slice = chosen_mask[:, start_col : start_col + L_x]
                
                # Randomly sample rows to match x.shape[0]
                M_rows = mask_slice.shape[0]
                row_indices = range(N_x)
                final_mask = mask_slice[row_indices, :]
                
                # Force masked values to 0.5
                x[final_mask == 1] = 0.5
                
                # --- Filter out invalid columns ---
                # Check which columns contain at least one 0
                has_zero = np.any(x == 0, axis=0)
                
                # Check which columns contain at least one 1
                has_one = np.any(x == 1, axis=0)
                
                # Keep only columns that have both
                cols_to_keep = has_zero & has_one
                
                # Overwrite x with the filtered array
                x = x[:, cols_to_keep]
                pos = pos[cols_to_keep]
            
            pos_diff = np.diff(pos, append = np.zeros(1))
            
            Y = []
            for k in range(y.shape[0]):
                a, b = ints[k]
                
                ii_ = np.where((pos * self.L >= a) & (pos * self.L < b))[0]
        
                Y.extend([y[k] for u in range(len(ii_))])

            Y = np.array(Y)            

            # Apply original transformations
            x = flip(x)
            
            with np.errstate(divide='ignore'):
                Y = np.log(Y)
            
            if len(self.scales) > 0:
                xw = pgml.compute_cwt(x, self.scales, config)
                x_concat = np.concatenate([np.expand_dims(x, 0), xw])
            else:
                x_concat = np.expand_dims(x, 0)
                
            x_concat = np.concatenate([x_concat, np.tile(pos_diff.reshape(1, 1, -1), (1, x_concat.shape[1], 1))], 0)
        
            max_idx = max(1, x_concat.shape[2] - self.args.size)
        
            for ij in range(self.args.n_samples):
                ii = np.random.randint(0, max_idx)
                
                X_chunk_np = x_concat[:, :, ii:ii + self.args.size]
                D_chunk_np = Y[ii:ii + self.args.size] 
                
                X_tensor = torch.from_numpy(X_chunk_np).float()
                D_tensor = torch.from_numpy(D_chunk_np).float()
                
                # Apply Internal Normalization
                if self.mean_x is not None and len(self.scales) > 0:
                    
            
                    mean_x_view = self.mean_x.view(-1, 1, 1)
                    std_x_view = self.std_x.view(-1, 1, 1)
                    
                    X_norm = X_tensor.clone()
                    X_norm[1:-1] = (X_tensor[1:-1] - mean_x_view) / std_x_view
                    X_tensor = X_norm
                
                if self.mean_y is not None:
                    valid_mask = torch.isfinite(D_tensor)
                    D_tensor = torch.where(
                        valid_mask, 
                        (D_tensor - self.mean_y) / self.std_y, 
                        D_tensor
                    )

                yield X_tensor, D_tensor + self.dilation
                
        # Clean up database handles when the epoch finishes
        for env in envs.values():
            env.close()

def compute_dataset_stats(dataloader, num_batches=100, factor = 2.):
    """Computes stats while masking out -inf diagonal values in targets."""
    sum_x, sum_x2 = None, None
    total_elements_x = 0
    sum_y, sum_y2 = None, None
    total_elements_y = 0
    
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    if global_rank == 0:
        logging.info(f"Estimating stats over {num_batches} batches...")
    
    for i, (X_batch, Y_batch) in enumerate(dataloader):
        if i >= num_batches: break
            
        # X Stats
        reduce_dims_x = [0] + list(range(2, X_batch.dim()))
        batch_sum_x = X_batch.sum(dim=reduce_dims_x)
        batch_sum_x2 = (X_batch ** 2).sum(dim=reduce_dims_x)
        
        if sum_x is None:
            sum_x = torch.zeros_like(batch_sum_x)
            sum_x2 = torch.zeros_like(batch_sum_x2)
            
        sum_x += batch_sum_x
        sum_x2 += batch_sum_x2
        
        elements_x = 1
        for dim in reduce_dims_x: elements_x *= X_batch.shape[dim]
        total_elements_x += elements_x

        # Y Stats (handling -inf)
        valid_mask = torch.isfinite(Y_batch)
        clean_y = torch.where(valid_mask, Y_batch, torch.tensor(0.0, dtype=Y_batch.dtype))
        
        batch_sum_y = clean_y.sum()
        batch_sum_y2 = (clean_y ** 2 * valid_mask.float()).sum()
        
        if sum_y is None:
            sum_y = torch.tensor(0.0, dtype=Y_batch.dtype)
            sum_y2 = torch.tensor(0.0, dtype=Y_batch.dtype)
            
        sum_y += batch_sum_y
        sum_y2 += batch_sum_y2
        total_elements_y += valid_mask.sum().item()

    mean_x = sum_x / total_elements_x
    var_x = (sum_x2 / total_elements_x) - (mean_x ** 2)
    std_x = torch.sqrt(torch.clamp(var_x, min=1e-8))
    
    mean_y = sum_y / total_elements_y
    var_y = (sum_y2 / total_elements_y) - (mean_y ** 2)
    std_y = torch.sqrt(torch.clamp(var_y, min=1e-8)) * factor
    
    return (mean_x[1:-1], std_x[1:-1]), (mean_y, std_y)

import sys
import json
import os
from datetime import datetime
import subprocess

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    args = parse_args()
    
    # Assuming you already have this part:
    # args = parse_args()
    
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    current_time = datetime.now().isoformat()
    
    # 1. Combine sys.argv and the parsed args into a single dictionary
    run_data = {
        "sys_argv": sys.argv,
        "parsed_args": vars(args), # vars() converts the Namespace to a dict
        "git" : git_hash,
        "timestamp" : current_time
    }
    
    # 2. Construct the file path using args.odir
    # You can name the file whatever you like (e.g., 'arguments.json')
    output_file = os.path.join(args.odir, "arguments.json")
    
    # 3. Ensure the output directory actually exists before writing
    os.makedirs(args.odir, exist_ok=True)
    
    # 4. Save the data to the JSON file
    with open(output_file, "w") as f:
        json.dump(run_data, f, indent=4)
    
    print(f"Arguments saved to {output_file}")
    #torch.multiprocessing.set_start_method('spawn', force=True)
    
    # Check if script was launched via torchrun
    is_distributed = "LOCAL_RANK" in os.environ
    
    if is_distributed:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        global_rank = 0
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # 1. Discover and shuffle all files
    data_dir = args.idir
    all_files = glob.glob(os.path.join(data_dir, "*", "*.npz"))
    
    # Use a fixed seed for shuffling so your split is reproducible across runs/resumes
    random.Random(42).shuffle(all_files)
    
    # 2. Split the files (e.g., 90% train, 10% validation)
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Found {len(all_files)} total files. Train: {len(train_files)}, Val: {len(val_files)}")
    
    if args.masks == "None":
        masks = None
    else:
        masks = args.masks
    
    # 3. Instantiate the datasets
    train_dataset = DiskWaveletDataset(args, train_files, n = args.n, two_pop = args.two_pop, dilation = args.dilation, L = args.L, 
                                       mask_folder = masks, return_rec_dist = args.rec)
    val_dataset = DiskWaveletDataset(args, val_files, n = args.n, two_pop = args.two_pop, dilation = args.dilation, 
                                     L = args.L, mask_folder = masks, return_rec_dist = args.rec)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        prefetch_factor=args.prefetch,
        pin_memory=(device.type == 'cuda'),
        worker_init_fn=worker_init_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        prefetch_factor=args.prefetch,
        pin_memory=(device.type == 'cuda'),
        worker_init_fn=worker_init_fn,
    )    
    
    model = ConvGCNUNet(
        in_dim=len(train_dataset.scales) + 2, 
        embedding_dim=args.embed_dim, 
        K=args.k, 
        dist=args.dist,
        return_rec_dist=args.rec
    ).to(device)
    
    # Load Weights (If applicable)
    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location=device)
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(filtered_dict, strict=False)
        
        if not args.recompute_means:
            stats = np.load(os.path.join('/'.join(args.weights.split('/')[:-1]), 'stats.npz'))
            
            mean_x = torch.FloatTensor(stats['mean_x'])
            std_x = torch.FloatTensor(stats['std_x'])
            mean_y = torch.FloatTensor(stats['mean_y'])
            std_y = torch.FloatTensor(stats['std_y'])
        else:
            num_stat_batches = 32
            (mean_x, std_x), (mean_y, std_y) = compute_dataset_stats(train_loader, num_batches=num_stat_batches, factor = args.factor)
            
        
    else:
        # 3. Compute and Inject Normalization Stats
        # (Uses fewer batches if testing on CPU)
        num_stat_batches = 32
        (mean_x, std_x), (mean_y, std_y) = compute_dataset_stats(train_loader, num_batches=num_stat_batches, factor = args.factor)
        
    np.savez(os.path.join(args.odir, 'stats.npz'), mean_x = mean_x, std_x = std_x, mean_y = mean_y, std_y = std_y)

    train_dataset.inject_stats(mean_x, std_x, mean_y, std_y)
    val_dataset.inject_stats(mean_x, std_x, mean_y, std_y)

    # 2. Wrap in DDP & Compile
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    
    # 4. Optimizer Setup
    if args.dist != 'poincare':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = RiemannianAdam(model.parameters(), lr=args.lr)
    
    criterion = nn.SmoothL1Loss()
    
    result = {'loss': [], 'time': [], 'epoch': [], 'val_loss' : []}
    batches_per_epoch = 2500 * args.n_batch
    best_val_loss = np.inf
    
    if global_rank == 0:
        logging.info("Starting Training...")
    
    # 5. Training Loop
    for epoch in range(1, args.epochs + 1):
    
        # ===============================
        #       TRAINING PHASE
        # ===============================
        model.train()
        running_train_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()
        
        # Pull a finite chunk out of the infinite generator
        total_samples = 0
        
        interval_loss = 0.0
        interval_samples = 0
        
        # You can add this to your args, defaulting to 100 here if it's missing
        log_interval = getattr(args, 'log_interval', 100)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs, _ = model(inputs)
            
            # Mask out -infs for the loss calculation
            valid_mask = torch.isfinite(targets)
            loss = criterion(outputs[valid_mask], targets[valid_mask])
        
            # Track losses
            batch_loss = loss.item() * inputs.size(0)
            running_train_loss += batch_loss
            interval_loss += batch_loss
            
            total_samples += inputs.size(0)
            interval_samples += inputs.size(0)
            
            loss = loss / args.n_batch
            loss.backward()
        
            if (batch_idx + 1) % args.n_batch == 0 or (batch_idx + 1) == batches_per_epoch:
                optimizer.step()
                optimizer.zero_grad()
                
            # --- Intra-epoch Logging ---
            if (batch_idx + 1) % log_interval == 0:
                if global_rank == 0:
                    avg_interval_loss = interval_loss / max(1, interval_samples)
                    logging.info(
                        f"Epoch: [{epoch:03d}/{args.epochs:03d}] | "
                        f"Batch: {batch_idx + 1:04d} | "
                        f"Interval Loss: {avg_interval_loss:.6f} | "
                        f"Elapsed Time: {time.time() - t0:.2f}s"
                    )
                # Reset interval trackers
                interval_loss = 0.0
                interval_samples = 0
                
            if batch_idx > batches_per_epoch:
                break
                    
        # Synchronize and Average Train Loss across GPUs
        epoch_train_loss = running_train_loss / max(1, total_samples)
        if is_distributed:
            loss_tensor = torch.tensor(epoch_train_loss).to(device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_train_loss = (loss_tensor / dist.get_world_size()).item()
    
        # ===============================
        #      VALIDATION PHASE
        # ===============================
        model.eval()
        running_val_loss = 0.0
        total_val_samples = 0
        
        # Pull a finite chunk for validation
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs, _ = model(inputs)
                
                valid_mask = torch.isfinite(targets)
                loss = criterion(outputs[valid_mask], targets[valid_mask])
                
                running_val_loss += loss.item() * inputs.size(0)
                total_val_samples += inputs.size(0)
                
        # Synchronize and Average Val Loss across GPUs
        epoch_val_loss = running_val_loss / max(1, total_val_samples)
        if is_distributed:
            val_loss_tensor = torch.tensor(epoch_val_loss).to(device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_val_loss = (val_loss_tensor / dist.get_world_size()).item()
    
        # ===============================
        #      LOGGING & CHECKPOINTING
        # ===============================
        # ONLY Rank 0 logs and saves checkpoints
        if global_rank == 0:
            result['loss'].append(epoch_train_loss)
            result['val_loss'].append(epoch_val_loss)
            result['time'].append(time.time() - t0)
            result['epoch'].append(epoch)
            
            df = pd.DataFrame(result)
            df.to_csv(os.path.join(args.odir, 'losses.csv'), index = False)
    
            # Check against validation loss, not training loss
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                
                if args.odir != "None":
                    save_path = os.path.join(args.odir, "best_model.pth")
                    # Safe DDP unwrapping
                    state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                    torch.save(state_dict, save_path)
                    logging.info(f"--> Loss improved. Saved checkpoint to {save_path}")
    
            logging.info(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | time: {result['time'][-1]:3f}")
    if is_distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()