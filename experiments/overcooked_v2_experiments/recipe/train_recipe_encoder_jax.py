import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state

from overcooked_v2_experiments.recipe.recipe_encoder_jax import (
    RecipeEncoder,
    RecipeNPZDataset,
)


def _split_indices(n: int, val_split: float, seed: int):
    if n <= 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n, dtype=np.int32)
    rng.shuffle(all_idx)

    val_size = int(n * val_split)
    if n >= 2:
        val_size = min(max(val_size, 1), n - 1)
    else:
        val_size = 0

    val_idx = all_idx[:val_size]
    train_idx = all_idx[val_size:]
    return train_idx, val_idx


def _iter_batches(dataset, indices, batch_size: int, shuffle: bool, seed: int):
    if len(indices) == 0:
        return

    idx = np.array(indices, copy=True)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for start in range(0, len(idx), batch_size):
        batch_ids = idx[start : start + batch_size]
        obs_list = []
        act_list = []
        label_list = []
        for i in batch_ids:
            obs, act, label = dataset[int(i)]
            obs_list.append(obs)
            act_list.append(act)
            label_list.append(label)

        yield (
            np.stack(obs_list, axis=0),
            np.stack(act_list, axis=0),
            np.array(label_list, dtype=np.int32),
        )


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

    train_idx, val_idx = _split_indices(len(full_dataset), val_split, seed)
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

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

    best_val_acc = -1.0
    checkpointer = ocp.PyTreeCheckpointer()

    for epoch in range(epochs):
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0

        train_total_batches = max(math.ceil(len(train_idx) / batch_size), 1)
        pbar = tqdm(
            _iter_batches(
                full_dataset,
                train_idx,
                batch_size=batch_size,
                shuffle=True,
                seed=seed + epoch,
            ),
            total=train_total_batches,
            desc=f"Epoch {epoch + 1}/{epochs} [Train]",
            leave=False,
        )
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

        for batch in _iter_batches(
            full_dataset,
            val_idx,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        ):
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
            segment_k = None
            dataset_meta = {}
            dataset_meta_path = os.path.join(os.path.abspath(data_dir), "dataset_meta.json")
            if os.path.exists(dataset_meta_path):
                try:
                    with open(dataset_meta_path, "r", encoding="utf-8") as df:
                        dataset_meta = json.load(df)
                    if "segment_k" in dataset_meta:
                        segment_k = int(dataset_meta["segment_k"])
                except Exception:
                    segment_k = None

            with open(meta_path, "w", encoding="utf-8") as f:
                meta = {
                    "use_actions": bool(use_actions),
                    "num_classes": int(full_dataset.num_classes),
                    "action_dim": int(action_dim),
                    "data_dir": os.path.abspath(data_dir),
                }
                if segment_k is not None:
                    meta["segment_k"] = int(segment_k)
                for key in [
                    "recipe_codes",
                    "collection_policy",
                    "obs_source",
                    "agent_0_role",
                    "agent_1_role",
                ]:
                    if key in dataset_meta:
                        meta[key] = dataset_meta[key]
                json.dump(meta, f, indent=2)
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
