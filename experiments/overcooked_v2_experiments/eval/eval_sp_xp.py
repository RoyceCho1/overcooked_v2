
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import os
import sys
import itertools

from overcooked_v2_experiments.ppo.run import load_fcp_populations
from overcooked_v2_experiments.ppo.policy import PPOPolicy, PPOParams
from overcooked_v2_experiments.eval.evaluate import eval_pairing
from overcooked_v2_experiments.eval.policy import PolicyPairing
from overcooked_v2_experiments.ppo.utils.store import load_checkpoint

class FCPPolicy(PPOPolicy):
    def compute_action(self, obs, done, hstate, key, params=None):
        if params is None:
            params = self.params
        assert params is not None

        done = jnp.array(done)

        def _add_dim(tree):
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], tree)

        # Handle Context Splitting
        # If obs has 40 channels, we assume 38 (Grid) + 2 (Context)
        # FCP network expects (obs, done, ctx) where obs is 38.
        if obs.shape[-1] == 40:
            real_obs = obs[..., :38]
            ctx = obs[..., 38:]
            ac_in = (real_obs, done, ctx)
        else:
            # Fallback for standard PPO or if obs is already 38
            ac_in = (obs, done)

        ac_in = _add_dim(ac_in)
        if not self.with_batching:
            ac_in = _add_dim(ac_in)

        # jax.debug.print("FCPPolicy ac_in shapes: {}", [x.shape for x in ac_in])

        next_hstate, pi, _ = self.network.apply(params, hstate, ac_in)

        if self.stochastic:
            action = pi.sample(seed=key)
        else:
            action = jnp.argmax(pi.probs, axis=-1)

        if self.with_batching:
            action = action[0]
        else:
            action = action[0, 0]

        return action, next_hstate

def load_fcp_policy(run_dir, run_num, checkpoint_idx):
    config, params = load_checkpoint(run_dir, run_num, checkpoint_idx)
    # Ensure FCP config flag is present if needed, though network structure dictates input
    return FCPPolicy(params=params, config=config)

def evaluate_fcp_population(fcp_base_dir, layout_name, num_episodes=10):
    print(f"Evaluating FCP Population at {fcp_base_dir}")
    print(f"Layout: {layout_name}")

    fcp_base_path = Path(fcp_base_dir)
    
    # 1. Load ALL FCP Agents
    fcp_policies = []
    # Assuming run_0 to run_7 exist
    # We can detect them
    run_dirs = sorted([d for d in fcp_base_path.iterdir() if d.is_dir() and d.name.startswith("run_")])
    
    if not run_dirs:
        print("No run_X directories found!")
        return

    print(f"Found {len(run_dirs)} FCP runs.")

    for run_dir in run_dirs:
        run_num = int(run_dir.name.split("_")[1])
        print(f"Loading FCP Agent from {run_dir.name} (ckpt_final)")
        policy = load_fcp_policy(fcp_base_path, run_num, "final")
        fcp_policies.append(policy)

    num_agents = len(fcp_policies)
    sp_scores = []
    xp_scores = []

    rng = jax.random.PRNGKey(42)

    print("\n--- Starting Evaluation Matrix ---")
    
    # We want to compute:
    # SP: Diagonal (Agent i vs Agent i)
    # XP: Off-diagonal (Agent i vs Agent j)
    
    # Note: PolicyPairing(agent0, agent1)
    # We should probably evaluate both (i, j) and (j, i) for XP?
    # Or just (i, j) and assume symmetry/average?
    # Let's do full matrix for completeness.

    matrix = np.zeros((num_agents, num_agents))

    for i in range(num_agents):
        for j in range(num_agents):
            agent_i = fcp_policies[i]
            agent_j = fcp_policies[j]
            
            pairing = PolicyPairing(agent_i, agent_j)
            
            rng, key = jax.random.split(rng)
            results = eval_pairing(
                policies=pairing,
                layout_name=layout_name,
                key=key,
                num_seeds=num_episodes,
                no_viz=True
            )
            
            mean_reward = np.mean([v.total_reward for v in results.values()])
            matrix[i, j] = mean_reward
            
            type_str = "SP" if i == j else "XP"
            print(f"[{type_str}] Agent {i} vs Agent {j}: {mean_reward:.2f}")

            if i == j:
                sp_scores.append(mean_reward)
            else:
                xp_scores.append(mean_reward)

    # Statistics
    sp_scores = np.array(sp_scores)
    xp_scores = np.array(xp_scores)
    
    # Gap calculation
    # Gap = Mean(SP) - Mean(XP)
    # Or Gap per agent? 
    # Let's report aggregate statistics.
    
    print("\n--- Final Results ---")
    print(f"SP Score: {np.mean(sp_scores):.2f} ± {np.std(sp_scores):.2f}")
    print(f"XP Score: {np.mean(xp_scores):.2f} ± {np.std(xp_scores):.2f}")
    
    gap = np.mean(sp_scores) - np.mean(xp_scores)
    # Std of gap? Maybe sqrt(std_sp^2 + std_xp^2) if independent, but they are correlated.
    # Simple difference of means is the main metric.
    print(f"Gap (SP - XP): {gap:.2f}")

@hydra.main(version_base=None, config_path="../ppo/config", config_name="base")
def main(config):
    fcp_run_dir = config.get("FCP_RUN_DIR", None)
    layout_name = config.env.ENV_KWARGS.layout
    
    if not fcp_run_dir:
        print("Please provide FCP_RUN_DIR (the base directory containing run_X folders)")
        return

    evaluate_fcp_population(fcp_run_dir, layout_name)

if __name__ == "__main__":
    main()
