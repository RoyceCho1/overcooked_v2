
import jax.numpy as jnp

def get_mask_fn(num_ingredients=3):
    """
    Return a function that masks recipe info from obs.
    This ensures that the Recipe Encoder only sees the observation
    channels that are allowed (no direct recipe leakage),
    matching the training distribution.
    
    Args:
        num_ingredients (int): Number of ingredients in the layout (default 3 for demo_cook_simple).
        
    Returns:
        mask_fn (callable): Function taking (..., H, W, C) obs and returning masked obs.
    """
    def mask_fn(obs):
        # obs: (..., H, W, C)
        
        # 1. Mask Recipe Layers (Dynamic Goal)
        # These are the channels that directly show the required recipe on the HUD/map.
        # They are located at [..., -1 - num_ingredients : -1] relative to the end (excluding extra info).
        # We assume 1 extra info layer (pot timer).
        obs = obs.at[..., -1 - num_ingredients : -1].set(0)
        
        # 2. Mask Static Indicators
        # For demo_cook_simple (38 channels), heuristics:
        # Static Object Indices for Recipe Indicator (19) and Button Indicator (20).
        # These markers on the map also reveal the recipe (e.g. "Order: Onion").
        obs = obs.at[..., 19].set(0)
        obs = obs.at[..., 20].set(0)
        
        return obs
    return mask_fn
