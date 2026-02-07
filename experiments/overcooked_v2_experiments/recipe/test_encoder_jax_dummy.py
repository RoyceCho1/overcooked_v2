import jax
import jax.numpy as jnp
from overcooked_v2_experiments.recipe.recipe_encoder_jax import RecipeEncoder

def test_model():
    print("Testing JAX RecipeEncoder forward pass...")
    
    # Dummy parameters
    B = 4
    K = 10
    H = 5
    W = 5
    C = 26
    num_actions = 6
    
    model = RecipeEncoder(num_actions=num_actions)
    
    key = jax.random.PRNGKey(0)
    
    # Create dummy input
    obs = jax.random.normal(key, (B, K, H, W, C))
    act = jax.random.normal(key, (B, K, num_actions))
    
    try:
        # Init
        variables = model.init(key, obs, act)
        print("Model initialized successfully.")
        
        # Apply
        logits = model.apply(variables, obs, act)
        print(f"Output shape: {logits.shape}")
        
        assert logits.shape == (B, 2)
        print("Test Passed: Output shape is correct.")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        raise e

if __name__ == "__main__":
    test_model()
