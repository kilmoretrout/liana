import argparse
import torch
from safetensors.torch import save_file

def convert_pt_to_safetensors(pt_path, out_path):
    # 1. Load the weights (force to CPU to avoid device mismatch issues)
    print(f"Loading {pt_path}...")
    state_dict = torch.load(pt_path, map_location="cpu")
    
    # If you used torch.save(model) instead of torch.save(model.state_dict()),
    # we need to extract just the state_dict dictionary.
    # If you used torch.save(model) instead of torch.save(model.state_dict()),
    # we need to extract just the state_dict dictionary.
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    elif hasattr(state_dict, "named_parameters"):
        state_dict = {k: v for k, v in state_dict.named_parameters()}
        
    new_state_dict = {}
    for k, v in state_dict.items():
        # If PyTorch used a Conv2d to simulate a Conv1d, the shape is [O, I, 1, K].
        # We squeeze dimension 2 to make it [O, I, K] for Candle's native Conv1d.
        if len(v.shape) == 4 and v.shape[2] == 1 and "conv" in k:
            v = v.squeeze(2)
            
        if k.startswith("lstm.bilstm."):
            is_reverse = "_reverse" in k
            is_l1 = "_l1" in k
            
            # Determine which of the 4 Rust LSTMs this belongs to
            direction = "bwd" if is_reverse else "fwd"
            layer = "l1" if is_l1 else "l0"
            rust_component = f"lstm_{direction}_{layer}"
            
            # Extract whether it's weight_ih, weight_hh, bias_ih, or bias_hh
            param_type = "weight_ih" if "weight_ih" in k else \
                         "weight_hh" if "weight_hh" in k else \
                         "bias_ih" if "bias_ih" in k else "bias_hh"
                         
            # Candle ALWAYS expects "_l0" for a single-layer LSTM instance
            k = f"lstm.{rust_component}.{param_type}_l0"

        # Clone breaks shared memory pointers, contiguous makes it safe for Rust
        new_state_dict[k] = v.clone().contiguous()

    # 2. Save to Safetensors
    print(f"Saving to {out_path}...")
    save_file(new_state_dict, out_path)
    print(f"Successfully converted and saved to {out_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch weights to Safetensors format for Rust.")
    
    parser.add_argument("-i", "--ifile", type=str, required=True, 
                        help="Input PyTorch weights file (e.g., model.pt)")
    parser.add_argument("-o", "--ofile", type=str, required=True, 
                        help="Output Safetensors file (e.g., model.safetensors)")
    
    args = parser.parse_args()
    
    convert_pt_to_safetensors(args.ifile, args.ofile)
