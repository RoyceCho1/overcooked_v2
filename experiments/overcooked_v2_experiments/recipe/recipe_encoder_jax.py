import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import os
import glob
from torch.utils.data import Dataset

class RecipeNPZDataset(Dataset):
    """
    Dataset for loading order-level recipe data from .npz files.
    Compatible with PyTorch DataLoader for easy batching and shuffling.
    """
    def __init__(self, data_dir):
        super().__init__()
        self.file_paths = glob.glob(os.path.join(data_dir, "partner_*.npz"))
        
        self.obs_list = []
        self.act_list = []
        self.label_list = []
        
        if not self.file_paths:
            print(f"Warning: No .npz files found in {data_dir}")
        else:
            print(f"Loading data from {len(self.file_paths)} files...")
            for path in self.file_paths:
                try:
                    data = np.load(path)
                    self.obs_list.append(data['obs'])
                    self.act_list.append(data['act'])
                    self.label_list.append(data['recipe'])
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            
            if self.obs_list:
                self.obs = np.concatenate(self.obs_list, axis=0)
                self.act = np.concatenate(self.act_list, axis=0)
                self.label = np.concatenate(self.label_list, axis=0)
            else:
                self.obs = np.array([])
                self.act = np.array([])
                self.label = np.array([])
                
            print(f"Total samples: {len(self.obs)}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        # Return numpy arrays. JAX handles numpy arrays fine.
        # obs: (K, H, W, C)
        obs = self.obs[idx]
        # act: (K, num_actions)
        act = self.act[idx]
        # label: scalar
        label = self.label[idx]
        
        return obs, act, label


class RecipeEncoder(nn.Module):
    """
    Recipe Encoder Network using Flax Linen.
    """
    num_actions: int = 6
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, obs, act):
        """
        Args:
            obs: (Batch, K, H, W, C)
            act: (Batch, K, num_actions)
        Returns:
            logits: (Batch, 2)
        """
        B, K, H, W, C = obs.shape
        
        # 1. CNN Feature Extractor
        # Apply to each step independently.
        # Flatten B and K: (B*K, H, W, C)
        x = obs.reshape((B * K, H, W, C))
        
        # Conv1: 5x5, padding=2 (SAME)
        x = nn.Conv(features=25, kernel_size=(5, 5), padding='SAME')(x)
        x = nn.leaky_relu(x)
        
        # Conv2: 3x3, padding=1 (SAME)
        x = nn.Conv(features=25, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.leaky_relu(x)
        
        # Conv3: 3x3, padding=0 (VALID) -> Matches PECAN's conv3 padding=(0,0) approx
        # Or just use SAME to keep dimensions simple?
        # PECAN: conv3 padding=(0,0).
        # Let's use VALID to reduce dimension slightly.
        x = nn.Conv(features=25, kernel_size=(3, 3), padding='VALID')(x)
        x = nn.leaky_relu(x)
        
        # Flatten spatial dims
        x = x.reshape((x.shape[0], -1))
        
        # State Embedding
        x = nn.Dense(features=64)(x)
        x = nn.leaky_relu(x)
        
        # 2. Concatenate with Action
        # act: (B, K, num_actions) -> (B*K, num_actions)
        act_flat = act.reshape((B * K, -1))
        x = jnp.concatenate([x, act_flat], axis=1)
        
        # Transition Embedding
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.leaky_relu(x)
        
        # 3. Reshape back to sequence
        x = x.reshape((B, K, self.hidden_dim))
        
        # 4. Self Attention
        # Using MultiHeadDotProductAttention with num_heads=1 to mimic simple self-attn
        x = nn.SelfAttention(num_heads=1, qkv_features=self.hidden_dim)(x)
        
        # 5. Temporal Aggregation (Sum over K)
        x = jnp.sum(x, axis=1) # (B, hidden_dim)
        
        # Trajectory Embedding
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.leaky_relu(x)
        
        # 6. Classifier
        logits = nn.Dense(features=2)(x) # (B, 2)
        
        return logits
