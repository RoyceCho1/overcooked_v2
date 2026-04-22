import glob
import os

import jax.numpy as jnp
import numpy as np
from flax import linen as nn

try:
    from torch.utils.data import Dataset
except Exception:
    # PPO only needs RecipeEncoder (Flax/JAX). Keep module importable even when
    # torch binaries are unavailable or incompatible in the runtime image.
    class Dataset:
        pass


class RecipeNPZDataset(Dataset):
    """Load recipe classification segments from `partner_*.npz` files."""

    def __init__(self, data_dir: str, use_actions: bool = False):
        super().__init__()
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "partner_*.npz")))
        self.use_actions = use_actions

        self.obs = np.array([])
        self.act = np.array([])
        self.label = np.array([])
        self.action_dim = 0
        self.num_classes = 0

        if not self.file_paths:
            print(f"Warning: No .npz files found in {data_dir}")
            return

        print(f"Loading data from {len(self.file_paths)} files...")

        obs_list = []
        label_list = []
        act_list = []
        act_mode = None

        for path in self.file_paths:
            try:
                data = np.load(path)
                obs = data["obs"].astype(np.float32)
                labels = data["recipe"].astype(np.int32)
                has_act = "act" in data

                if act_mode is None:
                    act_mode = has_act
                elif act_mode != has_act:
                    raise ValueError(
                        "Mixed dataset format detected: some files have `act` and others do not."
                    )

                obs_list.append(obs)
                label_list.append(labels)
                if has_act:
                    act_list.append(data["act"].astype(np.float32))
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if not obs_list:
            print("Warning: Failed to load any valid dataset files.")
            return

        self.obs = np.concatenate(obs_list, axis=0)
        self.label = np.concatenate(label_list, axis=0)

        if act_mode:
            self.act = np.concatenate(act_list, axis=0)
            self.action_dim = int(self.act.shape[-1])
        else:
            # Obs-only dataset: represent missing actions as a 0-width feature.
            seq_len = self.obs.shape[1]
            self.act = np.zeros((len(self.obs), seq_len, 0), dtype=np.float32)
            self.action_dim = 0

        self.num_classes = int(self.label.max()) + 1 if len(self.label) > 0 else 0

        if self.use_actions and self.action_dim == 0:
            raise ValueError(
                "use_actions=True but dataset has no `act` field. "
                "Set use_actions=False or recollect dataset with actions."
            )

        print(f"Total samples: {len(self.obs)}")
        print(f"Action dim: {self.action_dim}")
        print(f"Num classes: {self.num_classes}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx], self.label[idx]


class RecipeEncoder(nn.Module):
    """Temporal recipe classifier from ego observation sequence (optional partner actions)."""

    num_actions: int = 0
    hidden_dim: int = 128
    num_classes: int = 2
    use_actions: bool = False

    @nn.compact
    def __call__(self, obs, act=None):
        """
        Args:
            obs: (B, K, H, W, C)
            act: (B, K, A) or None
        Returns:
            logits: (B, num_classes)
        """
        B, K, H, W, C = obs.shape

        x = obs.reshape((B * K, H, W, C))

        x = nn.Conv(features=25, kernel_size=(5, 5), padding="SAME")(x)
        x = nn.leaky_relu(x)

        x = nn.Conv(features=25, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.leaky_relu(x)

        x = nn.Conv(features=25, kernel_size=(3, 3), padding="VALID")(x)
        x = nn.leaky_relu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=64)(x)
        x = nn.leaky_relu(x)

        if self.use_actions:
            if act is None:
                act = jnp.zeros((B, K, self.num_actions), dtype=obs.dtype)
            act_flat = act.reshape((B * K, -1))
            x = jnp.concatenate([x, act_flat], axis=-1)

        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.leaky_relu(x)

        x = x.reshape((B, K, self.hidden_dim))
        x = nn.SelfAttention(num_heads=1, qkv_features=self.hidden_dim)(x)

        x = jnp.sum(x, axis=1)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.leaky_relu(x)

        logits = nn.Dense(features=self.num_classes)(x)
        return logits
