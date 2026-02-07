import torch
import os
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from torch.utils.data import DataLoader, random_split
import orbax.checkpoint as ocp
import numpy as np

from overcooked_v2_experiments.recipe.recipe_encoder_jax import RecipeEncoder, RecipeNPZDataset

def train(
    data_dir="/home/myuser/recipe_data",
    save_path="/home/myuser/recipe_encoder_ckpt",
    batch_size=32,
    epochs=10,
    lr=1e-3,
    val_split=0.2,
    seed=0
):
    print(f"JAX devices: {jax.devices()}")
    
    # 1. Load Dataset
    full_dataset = RecipeNPZDataset(data_dir)
    
    if len(full_dataset) == 0:
        print("No data found. Exiting.")
        return

    # Split Train/Val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Use collate_fn=None (default) which stacks numpy arrays into torch tensors
    # BUT JAX prefers numpy arrays. 
    # We can use a custom collate_fn to return numpy arrays.
    def numpy_collate(batch):
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple,list)):
            transposed = zip(*batch)
            return [np.stack(samples) for samples in transposed]
        else:
            return np.array(batch)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)
    
    # 2. Initialize Model and State
    # Get a sample to determine input shape
    sample_obs, sample_act, _ = full_dataset[0]
    # sample_obs: (K, H, W, C)
    
    model = RecipeEncoder(num_actions=sample_act.shape[1])
    
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    
    # Add batch dim for init
    dummy_obs = jnp.array(sample_obs[None, ...])
    dummy_act = jnp.array(sample_act[None, ...])
    
    variables = model.init(init_key, dummy_obs, dummy_act)
    
    tx = optax.adam(learning_rate=lr)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )
    
    # 3. Define Train Step
    @jax.jit
    def train_step(state, obs, act, labels):
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, obs, act)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            return loss, logits
        
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        
        # Accuracy
        predicted_class = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predicted_class == labels)
        
        return state, loss, accuracy

    @jax.jit
    def eval_step(state, obs, act, labels):
        logits = state.apply_fn({'params': state.params}, obs, act)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        predicted_class = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predicted_class == labels)
        return loss, accuracy

    from tqdm import tqdm

    # 4. Training Loop
    best_val_acc = 0.0
    checkpointer = ocp.PyTreeCheckpointer()
    
    for epoch in range(epochs):
        # Train
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0
        
        # tqdm for training loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for batch in pbar:
            obs, act, labels = batch
            obs = jnp.array(obs)
            act = jnp.array(act)
            labels = jnp.array(labels)
            
            state, loss, acc = train_step(state, obs, act, labels)
            train_loss_sum += loss
            train_acc_sum += acc
            train_batches += 1
            
            # Update progress bar description with current loss/acc
            pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{acc:.2f}"})
            
        avg_train_loss = train_loss_sum / train_batches
        avg_train_acc = train_acc_sum / train_batches
        
        # Val
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0
        
        for batch in val_loader:
            obs, act, labels = batch
            obs = jnp.array(obs)
            act = jnp.array(act)
            labels = jnp.array(labels)
            
            loss, acc = eval_step(state, obs, act, labels)
            val_loss_sum += loss
            val_acc_sum += acc
            val_batches += 1
            
        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0
        avg_val_acc = val_acc_sum / val_batches if val_batches > 0 else 0.0
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}% | Val Acc: {avg_val_acc*100:.2f}%")
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            # Save checkpoint
            # Use PyTreeSave for compatibility
            save_args = ocp.args.PyTreeSave(state.params)
            
            abs_save_path = os.path.abspath(save_path)
            if os.path.exists(abs_save_path):
                import shutil
                shutil.rmtree(abs_save_path)
                
            checkpointer.save(abs_save_path, save_args)
            print(f"  -> Model saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/myuser/recipe_data")
    parser.add_argument("--save_path", type=str, default="/home/myuser/recipe_encoder_ckpt")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        save_path=args.save_path,
        epochs=args.epochs
    )
