import os
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from overcooked_v2_experiments.recipe.recipe_encoder_jax import RecipeEncoder, RecipeNPZDataset

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [np.stack(samples) for samples in transposed]
    else:
        return np.array(batch)

def evaluate_k(
    data_dir="/home/myuser/recipe_data_v2",
    ckpt_path="/home/myuser/recipe_encoder_ckpt_v2",
    k_values=[3, 5, 10],
    batch_size=128,
    seed=0
):
    print(f"JAX devices: {jax.devices()}")
    
    # 1. Load Dataset
    print(f"Loading data from {data_dir}...")
    full_dataset = RecipeNPZDataset(data_dir)
    if len(full_dataset) == 0:
        print("No data found.")
        return

    # Use 20% validation split
    val_split = 0.2
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=numpy_collate,
        num_workers=4
    )
    print(f"Validation set size: {len(val_dataset)} samples.")

    # 2. Load Model
    # Initialize with dummy data (max length 10)
    sample_obs, sample_act, _ = full_dataset[0]
    # sample_obs shape (10, 5, 5, 38)
    # Model expects (Batch, Time, ...)
    
    model = RecipeEncoder(num_actions=sample_act.shape[1])
    key = jax.random.PRNGKey(seed)
    
    # Init with full length to load params safely
    dummy_obs = jnp.zeros((1, 10, *sample_obs.shape[1:]))
    dummy_act = jnp.zeros((1, 10, sample_act.shape[1]))
    
    init_variables = model.init(key, dummy_obs, dummy_act)
    
    # Restore Checkpoint
    print(f"Restoring checkpoint from {ckpt_path}...")
    checkpointer = ocp.PyTreeCheckpointer()
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint {ckpt_path} not found!")
        return
        
    restored = checkpointer.restore(os.path.abspath(ckpt_path), item=init_variables['params'])
    params = restored

    @jax.jit
    def predict_step(params, obs, act):
        # Model forward
        # Assumes model handles variable length T if JIT-ed for that shape
        logits = model.apply({'params': params}, obs, act)
        return jnp.argmax(logits, axis=-1)

    print("\n=== Starting Evaluation for different K values ===")
    
    results = {}

    for k in k_values:
        print(f"\nEvaluating with K={k}...")
        all_preds = []
        all_labels = []
        
        # Re-JIT for new shape if needed (automatic in JAX if shape changes)
        
        for batch in val_loader:
            obs, act, labels = batch
            
            # Slice to K steps (taking first K)
            # obs: (B, 10, ...) -> (B, K, ...)
            obs_k = obs[:, :k]
            act_k = act[:, :k]
            
            obs_j = jnp.array(obs_k)
            act_j = jnp.array(act_k)
            
            preds = predict_step(params, obs_j, act_j)
            
            all_preds.extend(np.array(preds))
            all_labels.extend(np.array(labels))
            
        acc = accuracy_score(all_labels, all_preds)
        results[k] = acc
        print(f"Accuracy (K={k}): {acc*100:.2f}%")
        
        if k == 10: # Show full report for full length
            print("\nClassification Report (K=10):")
            print(classification_report(all_labels, all_preds))
            print("Confusion Matrix (K=10):")
            print(confusion_matrix(all_labels, all_preds))
            
    print("\n=== Summary ===")
    for k, acc in results.items():
        print(f"K={k}: {acc*100:.2f}%")

if __name__ == "__main__":
    evaluate_k()
