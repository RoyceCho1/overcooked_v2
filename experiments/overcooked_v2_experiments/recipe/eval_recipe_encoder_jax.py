
import os
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from overcooked_v2_experiments.recipe.recipe_encoder_jax import RecipeEncoder, RecipeNPZDataset

def evaluate(
    data_dir="/home/myuser/recipe_data",
    ckpt_path="/home/myuser/recipe_encoder_ckpt",
    batch_size=32,
    seed=0
):
    print(f"JAX devices: {jax.devices()}")
    
    # 1. Load Dataset
    # In a real scenario, you should have a separate test set.
    # Here we will load the same dataset and use the validation split (or all of it if intended)
    # Let's assume we want to evaluate on the validation set used during training to confirm performance,
    # or if there are new files, we could load those.
    # For now, let's replicate the split logic to get the validation set.
    
    full_dataset = RecipeNPZDataset(data_dir)
    if len(full_dataset) == 0:
        print("No data found.")
        return

    val_split = 0.2
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)) 
    # Note: torch random_split seed might differ if not set explicitly same as training.
    # Ideally, save test indices. But for quick check, this is fine.
    
    # Or just evaluate on EVERYTHING to see overall performance?
    # Let's evaluate on the validation set to be fair.
    
    def numpy_collate(batch):
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple,list)):
            transposed = zip(*batch)
            return [np.stack(samples) for samples in transposed]
        else:
            return np.array(batch)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)
    print(f"Evaluating on {len(val_dataset)} samples.")

    # 2. Load Model Checkpoint
    # Need to initialize model to get structure
    sample_obs, sample_act, _ = full_dataset[0]
    model = RecipeEncoder(num_actions=sample_act.shape[1])
    
    key = jax.random.PRNGKey(seed)
    dummy_obs = jnp.array(sample_obs[None, ...])
    dummy_act = jnp.array(sample_act[None, ...])
    
    # Init variables structure
    init_variables = model.init(key, dummy_obs, dummy_act)
    
    # Restore
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(os.path.abspath(ckpt_path), item=init_variables['params'])
    
    params = restored
    
    # 3. Evaluation Loop
    all_preds = []
    all_labels = []
    
    @jax.jit
    def predict_step(params, obs, act):
        logits = model.apply({'params': params}, obs, act)
        return jnp.argmax(logits, axis=-1)

    for batch in val_loader:
        obs, act, labels = batch
        obs = jnp.array(obs)
        act = jnp.array(act)
        
        preds = predict_step(params, obs, act)
        
        all_preds.extend(np.array(preds))
        all_labels.extend(np.array(labels))
        
    # 4. Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = np.mean(all_preds == all_labels)
    print(f"\nOverall Accuracy: {acc*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Recipe 0", "Recipe 1"]))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Optional: Save confusion matrix plot
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.title('Confusion Matrix')
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')
    # plt.savefig('confusion_matrix.png')
    # print("Saved confusion_matrix.png")

if __name__ == "__main__":
    import argparse
    import torch # for generator seeding
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/myuser/recipe_data")
    parser.add_argument("--ckpt_path", type=str, default="/home/myuser/recipe_encoder_ckpt")
    args = parser.parse_args()
    
    evaluate(args.data_dir, args.ckpt_path)
