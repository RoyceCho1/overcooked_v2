from functools import partial
import jax


def _check_and_return_outer_dim(args, kwargs, num_mini_batches):
    outer_dim = jax.tree_util.tree_flatten((args, kwargs))[0][0].shape[0]
    assert (
        outer_dim % num_mini_batches == 0
    ), f"outer_dim {outer_dim} must be divisible by num_mini_batches {num_mini_batches}"
    return outer_dim


def scanned_mini_batch_map(f, num_mini_batches, use_pmap=False, num_devices=None):
    """
    Execute a function in sequential, vmapped mini-batches.
    Enables execution of batches too large to fit in memory.
    """

    map_fn = jax.pmap if use_pmap else jax.vmap
    if num_devices:
        map_fn = partial(mini_batch_pmap, num_mini_batches=num_devices)

    def mapped_fn(*args, **kwargs):
        outer_dim = _check_and_return_outer_dim(args, kwargs, num_mini_batches)
        if outer_dim == num_mini_batches:
            return map_fn(f)(*args, **kwargs)

        def _batched_fn(_, x):
            x_args, x_kwargs = x
            print(f"Args: {x_args}")
            print(f"Kwargs: {x_kwargs}")
            y = map_fn(f)(*x_args, **x_kwargs)
            return None, y

        mini_batched_args, mini_batched_kwargs = jax.tree_map(
            lambda x: x.reshape((num_mini_batches, -1, *x.shape[1:])), (args, kwargs)
        )
        _, ret = jax.lax.scan(
            _batched_fn, None, (mini_batched_args, mini_batched_kwargs)
        )
        return jax.tree_map(lambda x: x.reshape((outer_dim, *x.shape[2:])), ret)

    return mapped_fn


def mini_batch_pmap(f, num_mini_batches):
    def mapped_fn(*args, **kwargs):
        outer_dim = _check_and_return_outer_dim(args, kwargs, num_mini_batches)
        if outer_dim == num_mini_batches:
            return jax.pmap(f)(*args, **kwargs)

        mini_batched_args, mini_batched_kwargs = jax.tree_map(
            lambda x: x.reshape((num_mini_batches, -1, *x.shape[1:])), (args, kwargs)
        )

        ret = jax.pmap(jax.vmap(f))(*mini_batched_args, **mini_batched_kwargs)

        return jax.tree_map(lambda x: x.reshape((outer_dim, *x.shape[2:])), ret)

    return mapped_fn
