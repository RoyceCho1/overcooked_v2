import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import torch
from flax.training import train_state
from torch.utils.data import DataLoader, random_split

from overcooked_v2_experiments.recipe.recipe_encoder_jax import (
    RecipeEncoder,
    RecipeNPZDataset,
)


def _numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    if isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [np.stack(samples) for samples in transposed]
    return np.array(batch)


def train(
    data_dir="./runs/recipe_data_demo_cook_simple_obs_only",
    save_path="./runs/recipe_encoder_ckpt_demo_cook_simple_obs_only",
    use_actions=False,
    batch_size=32,
    epochs=10,
    lr=1e-3,
    val_split=0.2,
    seed=0,
):
    print(f"JAX devices: {jax.devices()}")

    full_dataset = RecipeNPZDataset(data_dir, use_actions=use_actions)
    if len(full_dataset) == 0:
        print("No data found. Exiting.")
        return

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_numpy_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_numpy_collate,
    )

    sample_obs, sample_act, _ = full_dataset[0]

    action_dim = full_dataset.action_dim if use_actions else 0
    model = RecipeEncoder(
        num_actions=action_dim,
        num_classes=full_dataset.num_classes,
        use_actions=use_actions,
    )

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)

    dummy_obs = jnp.array(sample_obs[None, ...])
    if use_actions:
        dummy_act = jnp.array(sample_act[None, ...])
        variables = model.init(init_key, dummy_obs, dummy_act)
    else:
        variables = model.init(init_key, dummy_obs)

    tx = optax.adam(learning_rate=lr)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )

    @jax.jit
    def train_step(state, obs, act, labels):
        def loss_fn(params):
            if use_actions:
                logits = state.apply_fn({"params": params}, obs, act)
            else:
                logits = state.apply_fn({"params": params}, obs)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)

        predicted_class = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predicted_class == labels)

        return state, loss, accuracy

    @jax.jit
    def eval_step(state, obs, act, labels):
        if use_actions:
            logits = state.apply_fn({"params": state.params}, obs, act)
        else:
            logits = state.apply_fn({"params": state.params}, obs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        predicted_class = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predicted_class == labels)
        return loss, accuracy

    from tqdm import tqdm

    best_val_acc = 0.0
    checkpointer = ocp.PyTreeCheckpointer()

    for epoch in range(epochs):
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)
        for batch in pbar:
            obs, act, labels = batch
            obs = jnp.array(obs)
            act = jnp.array(act)
            labels = jnp.array(labels)

            state, loss, acc = train_step(state, obs, act, labels)
            train_loss_sum += float(loss)
            train_acc_sum += float(acc)
            train_batches += 1

            pbar.set_postfix({"loss": f"{float(loss):.4f}", "acc": f"{float(acc):.2f}"})

        avg_train_loss = train_loss_sum / max(train_batches, 1)
        avg_train_acc = train_acc_sum / max(train_batches, 1)

        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_batches = 0

        for batch in val_loader:
            obs, act, labels = batch
            obs = jnp.array(obs)
            act = jnp.array(act)
            labels = jnp.array(labels)

            loss, acc = eval_step(state, obs, act, labels)
            val_loss_sum += float(loss)
            val_acc_sum += float(acc)
            val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_acc = val_acc_sum / max(val_batches, 1)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {avg_train_acc * 100:.2f}% | "
            f"Val Acc: {avg_val_acc * 100:.2f}%"
        )

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_args = ocp.args.PyTreeSave(state.params)

            abs_save_path = os.path.abspath(save_path)
            if os.path.exists(abs_save_path):
                import shutil

                shutil.rmtree(abs_save_path)

            checkpointer.save(abs_save_path, save_args)
            print(f"  -> Model saved to {save_path}")

            meta_path = f"{abs_save_path}.meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "use_actions": bool(use_actions),
                        "num_classes": int(full_dataset.num_classes),
                        "action_dim": int(action_dim),
                        "data_dir": os.path.abspath(data_dir),
                    },
                    f,
                    indent=2,
                )
            print(f"  -> Metadata saved to {meta_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./runs/recipe_data_demo_cook_simple_obs_only")
    parser.add_argument(
        "--save_path",
        type=str,
        default="./runs/recipe_encoder_ckpt_demo_cook_simple_obs_only",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use_actions",
        action="store_true",
        help="Use partner action sequence as encoder input in addition to ego observation.",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        save_path=args.save_path,
        use_actions=args.use_actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
