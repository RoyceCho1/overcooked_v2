import argparse
import csv
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from overcooked_v2_experiments.recipe.recipe_encoder_jax import (
    RecipeEncoder,
    RecipeNPZDataset,
)


def _load_json_if_exists(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _split_indices(n: int, split: str, val_split: float, seed: int):
    indices = np.arange(n, dtype=np.int32)
    if split == "all":
        return indices

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = int(n * val_split)
    if n >= 2:
        val_size = min(max(val_size, 1), n - 1)
    else:
        val_size = 0

    if split == "val":
        return indices[:val_size]
    if split == "train":
        return indices[val_size:]
    raise ValueError(f"Unknown split: {split}")


def _iter_batches(dataset, indices, batch_size: int):
    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start : start + batch_size]
        obs_list = []
        act_list = []
        labels = []
        for idx in batch_ids:
            obs, act, label = dataset[int(idx)]
            obs_list.append(obs)
            act_list.append(act)
            labels.append(label)
        yield (
            np.stack(obs_list, axis=0),
            np.stack(act_list, axis=0),
            np.asarray(labels, dtype=np.int32),
        )


def _confusion_matrix(labels, preds, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y, p in zip(labels.tolist(), preds.tolist()):
        if 0 <= int(y) < num_classes and 0 <= int(p) < num_classes:
            cm[int(y), int(p)] += 1
    return cm


def _class_accuracy(cm):
    acc = []
    for cls in range(cm.shape[0]):
        denom = int(cm[cls].sum())
        value = float(cm[cls, cls] / denom) if denom else float("nan")
        acc.append(value)
    return acc


def _write_summary_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _write_summary_csv(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"metric": "direct_accuracy", "class": -1, "value": payload["direct_accuracy"], "n": payload["num_samples"]},
        {"metric": "swapped_accuracy", "class": -1, "value": payload["swapped_accuracy"], "n": payload["num_samples"]},
        {"metric": "direct_minus_swapped_accuracy", "class": -1, "value": payload["direct_minus_swapped_accuracy"], "n": payload["num_samples"]},
        {"metric": "label_mismatch_suspected", "class": -1, "value": float(payload["label_mismatch_suspected"]), "n": payload["num_samples"]},
    ]
    for cls, value in enumerate(payload["class_accuracy"]):
        rows.append({"metric": "class_accuracy", "class": cls, "value": value, "n": payload["class_counts"][cls]})

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "class", "value", "n"])
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    data_dir: Path,
    ckpt_path: Path,
    batch_size: int,
    seed: int,
    split: str,
    val_split: float,
    use_actions: bool,
    output_json: Path | None,
    output_csv: Path | None,
):
    dataset_meta = _load_json_if_exists(data_dir / "dataset_meta.json")
    encoder_meta = _load_json_if_exists(Path(str(ckpt_path) + ".meta.json"))
    recipe_codes = dataset_meta.get("recipe_codes", encoder_meta.get("recipe_codes", []))

    print(f"JAX devices: {jax.devices()}")
    print(f"data_dir: {data_dir}")
    print(f"ckpt_path: {ckpt_path}")
    print(f"split: {split}")
    print(f"dataset_meta.recipe_codes: {recipe_codes}")
    if dataset_meta:
        print(f"dataset_meta.collection_policy: {dataset_meta.get('collection_policy', 'unknown')}")
        print(f"dataset_meta.obs_source: {dataset_meta.get('obs_source', 'unknown')}")

    dataset = RecipeNPZDataset(str(data_dir), use_actions=use_actions)
    if len(dataset) == 0:
        raise ValueError(f"No data found in {data_dir}")

    indices = _split_indices(len(dataset), split=split, val_split=val_split, seed=seed)
    if len(indices) == 0:
        raise ValueError(f"No samples selected for split={split}")

    sample_obs, sample_act, _ = dataset[0]
    action_dim = dataset.action_dim if use_actions else 0
    num_classes = int(dataset.num_classes)
    model = RecipeEncoder(
        num_actions=action_dim,
        num_classes=num_classes,
        use_actions=use_actions,
    )

    dummy_obs = jnp.asarray(sample_obs[None, ...])
    if use_actions:
        dummy_act = jnp.asarray(sample_act[None, ...])
        dummy_params = model.init(jax.random.PRNGKey(seed), dummy_obs, dummy_act)["params"]
    else:
        dummy_params = model.init(jax.random.PRNGKey(seed), dummy_obs)["params"]

    params = ocp.PyTreeCheckpointer().restore(os.path.abspath(ckpt_path), item=dummy_params)

    @jax.jit
    def predict_step(obs, act):
        if use_actions:
            logits = model.apply({"params": params}, obs, act)
        else:
            logits = model.apply({"params": params}, obs)
        probs = jax.nn.softmax(logits, axis=-1)
        return jnp.argmax(probs, axis=-1), probs

    all_labels = []
    all_preds = []
    all_prob_true = []
    all_max_prob = []
    for obs, act, labels in _iter_batches(dataset, indices, batch_size):
        preds, probs = predict_step(jnp.asarray(obs), jnp.asarray(act))
        preds = np.asarray(preds, dtype=np.int32)
        probs = np.asarray(probs, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        all_labels.append(labels)
        all_preds.append(preds)
        all_prob_true.append(probs[np.arange(len(labels)), labels])
        all_max_prob.append(np.max(probs, axis=-1))

    labels = np.concatenate(all_labels, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    prob_true = np.concatenate(all_prob_true, axis=0)
    max_prob = np.concatenate(all_max_prob, axis=0)

    direct_correct = preds == labels
    direct_accuracy = float(np.mean(direct_correct))

    if num_classes == 2:
        swapped_preds = 1 - preds
        swapped_accuracy = float(np.mean(swapped_preds == labels))
    else:
        swapped_accuracy = float("nan")

    direct_minus_swapped = (
        direct_accuracy - swapped_accuracy
        if np.isfinite(swapped_accuracy)
        else float("nan")
    )
    label_mismatch_suspected = bool(
        num_classes == 2
        and np.isfinite(direct_minus_swapped)
        and direct_minus_swapped < -0.10
    )

    cm = _confusion_matrix(labels, preds, num_classes)
    class_acc = _class_accuracy(cm)
    class_counts = [int(np.sum(labels == cls)) for cls in range(num_classes)]

    payload = {
        "data_dir": str(data_dir),
        "ckpt_path": str(ckpt_path),
        "split": split,
        "seed": int(seed),
        "val_split": float(val_split),
        "use_actions": bool(use_actions),
        "num_samples": int(len(labels)),
        "num_classes": int(num_classes),
        "recipe_codes": [int(x) for x in recipe_codes],
        "direct_accuracy": direct_accuracy,
        "swapped_accuracy": swapped_accuracy,
        "direct_minus_swapped_accuracy": direct_minus_swapped,
        "label_mismatch_suspected": label_mismatch_suspected,
        "class_accuracy": class_acc,
        "class_counts": class_counts,
        "confusion_matrix": cm.tolist(),
        "mean_prob_true_recipe": float(np.mean(prob_true)),
        "mean_max_prob": float(np.mean(max_prob)),
    }

    print("\n=== Offline Recipe Encoder Sanity ===")
    print(f"num_samples: {payload['num_samples']}")
    print(f"num_classes: {payload['num_classes']}")
    print(f"recipe_codes: {payload['recipe_codes']}")
    print(f"direct_accuracy: {direct_accuracy:.6f}")
    print(f"swapped_accuracy: {swapped_accuracy:.6f}")
    print(f"direct_minus_swapped_accuracy: {direct_minus_swapped:.6f}")
    print(f"label_mismatch_suspected: {int(label_mismatch_suspected)}")
    print(f"mean_prob_true_recipe: {payload['mean_prob_true_recipe']:.6f}")
    print(f"mean_max_prob: {payload['mean_max_prob']:.6f}")
    print("\nClass accuracy:")
    for cls, value in enumerate(class_acc):
        print(f"  class {cls} (recipe_code={recipe_codes[cls] if cls < len(recipe_codes) else 'unknown'}): {value:.6f} n={class_counts[cls]}")
    print("\nConfusion matrix rows=true, cols=pred:")
    print(cm)

    if output_json is not None:
        _write_summary_json(output_json, payload)
        print(f"\nSaved JSON summary: {output_json}")
    if output_csv is not None:
        _write_summary_csv(output_csv, payload)
        print(f"Saved CSV summary: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--ckpt_path", required=True, type=Path)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["all", "train", "val"], default="val")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--use_actions", action="store_true")
    parser.add_argument("--output_json", default=None, type=Path)
    parser.add_argument("--output_csv", default=None, type=Path)
    args = parser.parse_args()

    evaluate(
        data_dir=args.data_dir,
        ckpt_path=args.ckpt_path,
        batch_size=args.batch_size,
        seed=args.seed,
        split=args.split,
        val_split=args.val_split,
        use_actions=args.use_actions,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
