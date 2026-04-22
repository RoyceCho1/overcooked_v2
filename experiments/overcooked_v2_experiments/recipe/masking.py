
import jax.numpy as jnp


def _compute_recipe_leakage_channels(
    num_ingredients: int,
    include_static_indicators: bool = True,
):
    """
    Compute recipe leakage channels for OvercookedV2 DEFAULT observation.

    Observation layout in `OvercookedV2.get_obs_default`:
      [agent_layer, other_agent_layers, static_layers, ingredient_pile_layers,
       ingredients_layers, recipe_layers, extra_layers]
    """
    ingredient_layer_width = 2 + num_ingredients
    agent_layer_width = 1 + 4 + ingredient_layer_width

    static_start = 2 * agent_layer_width
    ingredient_pile_start = static_start + 6
    ingredients_start = ingredient_pile_start + num_ingredients
    recipe_start = ingredients_start + ingredient_layer_width
    recipe_end = recipe_start + ingredient_layer_width

    indices = list(range(recipe_start, recipe_end))
    if include_static_indicators:
        indices.append(static_start + 3)  # RECIPE_INDICATOR
        indices.append(static_start + 4)  # BUTTON_RECIPE_INDICATOR

    return sorted(set(indices))


def get_mask_fn(
    num_ingredients: int,
    include_static_indicators: bool = True,
):
    """Return a JAX-compatible function masking recipe leakage channels."""
    leakage_channels = _compute_recipe_leakage_channels(
        num_ingredients=num_ingredients,
        include_static_indicators=include_static_indicators,
    )

    def mask_fn(obs):
        num_channels = obs.shape[-1]
        valid_channels = [idx for idx in leakage_channels if idx < num_channels]
        if not valid_channels:
            return obs

        channel_mask = jnp.ones((num_channels,), dtype=obs.dtype)
        channel_mask = channel_mask.at[jnp.array(valid_channels, dtype=jnp.int32)].set(0)
        return obs * channel_mask

    return mask_fn
