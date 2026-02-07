
import os
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
from overcooked_v2_experiments.ppo.run import load_fcp_populations

def verify_population(population_dir):
    print(f"Verifying population at: {population_dir}")
    path = Path(population_dir)
    
    if not path.exists():
        print(f"Error: Directory {population_dir} does not exist.")
        return

    # Check for run_X directories
    run_dirs = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith("run_")])
    print(f"Found {len(run_dirs)} run directories.")
    
    if len(run_dirs) == 0:
        print("Warning: No 'run_X' directories found. Is this the correct path?")
        return

    valid_runs = 0
    for run_dir in run_dirs:
        ckpt_path = run_dir / "ckpt_final"
        if ckpt_path.exists():
            valid_runs += 1
            # print(f"  {run_dir.name}: ckpt_final found.")
        else:
            print(f"  {run_dir.name}: ckpt_final MISSING!")

    print(f"Summary: {valid_runs}/{len(run_dirs)} runs have 'ckpt_final'.")

    if valid_runs > 0:
        print("\nAttempting to load population params...")
        try:
            population_params, config = load_fcp_populations(path)
            # population_params is a Ptree of stacked params
            
            # Inspect first layer kernel to verify input channels
            # Expected path: params -> CNN_0 -> Conv_0 -> kernel
            # But since it's stacked, it might be params['CNN_0']['Conv_0']['kernel'] with shape (N, H, W, In, Out)
            
            try:
                # Flatten to find the first kernel
                flat_params = jax.tree_util.tree_flatten_with_path(population_params)[0]
                
                conv0_kernel = None
                for path_tuple, value in flat_params:
                    # path_tuple is a tuple of DictKey/SequenceKey/GetAttrKey
                    keys = []
                    for p in path_tuple:
                        if hasattr(p, 'key'): keys.append(str(p.key))
                        elif hasattr(p, 'name'): keys.append(str(p.name))
                        elif hasattr(p, 'idx'): keys.append(str(p.idx))
                        else: keys.append(str(p))
                    path_str = "/".join(keys)
                    
                    if "Conv_0" in path_str and "kernel" in path_str:
                        conv0_kernel = value
                        print(f"Found Conv_0 kernel at: {path_str}")
                        break
                
                if conv0_kernel is not None:
                    # Shape: (Num_Pops, Num_Policies, H, W, In_Channels, Out_Channels)
                    # e.g. (1, 16, 5, 5, 38, 32)
                    shape = conv0_kernel.shape
                    print(f"Conv_0 Kernel Shape: {shape}")
                    input_channels = shape[-2]
                    print(f"Detected Input Channels: {input_channels}")
                    
                    if input_channels == 38:
                        print("✅ CORRECT: Input channels match demo_cook_simple (38).")
                    elif input_channels == 26:
                        print("❌ INCORRECT: Input channels match 5x5 layout (26).")
                    else:
                        print(f"⚠️ UNKNOWN: Input channels {input_channels} (Expected 38 for demo_cook_simple).")
                else:
                    print("Could not find Conv_0 kernel to verify shape.")

            except Exception as e:
                print(f"Error inspecting params: {e}")
                import traceback
                traceback.print_exc()

            leaves = jax.tree_util.tree_leaves(population_params)
            if leaves:
                # leaves[0] shape is (Num_Pops, Num_Policies, ...)
                # e.g. (1, 16, ...)
                shape = leaves[0].shape
                N = shape[1] if len(shape) > 1 else shape[0]
                print(f"Successfully loaded params for {N} agents (Population Count: {shape[0]}).")
                print("Population verification PASSED.")
            else:
                print("Loaded params are empty?")
        except Exception as e:
            print(f"Error loading population: {e}")
            print("Population verification FAILED.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m overcooked_v2_experiments.eval.verify_population <path_to_population_dir>")
        sys.exit(1)
    
    verify_population(sys.argv[1])
