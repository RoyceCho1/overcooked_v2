import functools
from typing import Dict, Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal
from .abstract import ActorCriticBase
from .common import CNN, MLP


class ScannedRNN(nn.Module):
    # train: bool = False

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x

        # print("rnn shapes new", rnn_state.shape, ins.shape, resets.shape)

        # resets = jax.lax.stop_gradient(resets)
        # new_carry = jax.lax.stop_gradient(
        #     self.initialize_carry(ins.shape[0], ins.shape[1])
        # )
        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])

        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            new_carry,
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)

        # y = nn.BatchNorm(use_running_average=not self.train)(y)

        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(ActorCriticBase):

    @nn.compact
    def __call__(self, hidden, x, train=False):
        obs, dones = x

        # print("cnn shapes", hidden.shape, obs.shape, dones.shape)

        embedding = obs

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # embedding_mlp = MLP(
        #     hidden_size=self.config["FC_DIM_SIZE"],
        #     output_size=self.config["FC_DIM_SIZE"],
        #     activation=activation,
        # )

        # embedding = jax.vmap(
        #     jax.vmap(embedding_mlp, in_axes=-2, out_axes=-2), in_axes=-2, out_axes=-2
        # )(embedding)

        embed_model = CNN(
            # output_size=self.config["FC_DIM_SIZE"] * 2,
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(embedding)

        embedding = nn.LayerNorm()(embedding)

        # embedding_1_mlp = MLP(
        #     hidden_size=self.config["FC_DIM_SIZE"],
        #     output_size=self.config["FC_DIM_SIZE"],
        #     activation=activation,
        # )
        # embedding = jax.vmap(
        #     jax.vmap(embedding_1_mlp, in_axes=-2, out_axes=-2), in_axes=-2, out_axes=-2
        # )(embedding)

        # embedding = nn.Dense(
        #     self.config["GRU_HIDDEN_DIM"],
        #     kernel_init=orthogonal(jnp.sqrt(2)),
        #     bias_init=constant(0.0),
        # )(embedding)

        rnn_in = (embedding, dones)
        # print("rnn_in shapes", hidden.shape, rnn_in[0].shape, rnn_in[1].shape)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # embedding = nn.Dense(
        #     self.config["FC_DIM_SIZE"],
        #     kernel_init=orthogonal(jnp.sqrt(2)),
        #     bias_init=constant(0.0),
        # )(embedding)

        # embedding_2_mlp = MLP(
        #     hidden_size=self.config["FC_DIM_SIZE"],
        #     output_size=self.config["FC_DIM_SIZE"],
        #     activation=activation,
        # )
        # embedding = jax.vmap(
        #     jax.vmap(embedding_2_mlp, in_axes=-2, out_axes=-2), in_axes=-2, out_axes=-2
        # )(embedding)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        # actor_mean = nn.BatchNorm(use_running_average=not self.train)(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        # critic = nn.BatchNorm(use_running_average=not self.train)(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        # print(
        #     "output shapes",
        #     hidden.shape,
        #     pi.logits.shape,
        #     jnp.squeeze(critic, axis=-1).shape,
        # )

        return hidden, pi, jnp.squeeze(critic, axis=-1)
