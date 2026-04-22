import os

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split

from overcooked_v2_experiments.recipe.recipe_encoder_jax import (
    RecipeEncoder,
    RecipeNPZDataset,
)


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    if isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [np.stack(samples) for samples in transposed]
    return np.array(batch)


def evaluate(
    data_dir="./runs/recipe_data_demo_cook_simple_obs_only",
    ckpt_path="./runs/recipe_encoder_ckpt_demo_cook_simple_obs_only",
    batch_size=32,
    seed=0,
    use_actions=False,
):
    print(f"JAX devices: {jax.devices()}")

    full_dataset = RecipeNPZDataset(data_dir, use_actions=use_actions)
    if len(full_dataset) == 0:
        print("No data found.")
        return

    val_split = 0.2
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=numpy_collate,
    )
    print(f"Evaluating on {len(val_dataset)} samples.")

    sample_obs, sample_act, _ = full_dataset[0]
    action_dim = full_dataset.action_dim if use_actions else 0
    model = RecipeEncoder(
        num_actions=action_dim,
        num_classes=full_dataset.num_classes,
        use_actions=use_actions,
    )

    key = jax.random.PRNGKey(seed)
    dummy_obs = jnp.array(sample_obs[None, ...])
    if use_actions:
        dummy_act = jnp.array(sample_act[None, ...])
        init_variables = model.init(key, dummy_obs, dummy_act)
    else:
        init_variables = model.init(key, dummy_obs)

    checkpointer = ocp.PyTreeCheckpointer()
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    params = checkpointer.restore(os.path.abspath(ckpt_path), item=init_variables["params"])

    @jax.jit
    def predict_step(params, obs, act):
        if use_actions:
            logits = model.apply({"params": params}, obs, act)
        else:
            logits = model.apply({"params": params}, obs)
        return jnp.argmax(logits, axis=-1)

    all_preds = []
    all_labels = []

    for batch in val_loader:
        obs, act, labels = batch
        preds = predict_step(params, jnp.array(obs), jnp.array(act))
        all_preds.extend(np.array(preds))
        all_labels.extend(np.array(labels))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = np.mean(all_preds == all_labels)
    print(f"\nOverall Accuracy: {acc * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./runs/recipe_data_demo_cook_simple_obs_only")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./runs/recipe_encoder_ckpt_demo_cook_simple_obs_only",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_actions", action="store_true")
    args = parser.parse_args()

    evaluate(
        data_dir=args.data_dir,
        ckpt_path=args.ckpt_path,
        batch_size=args.batch_size,
        use_actions=args.use_actions,
    )
