# train.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import itertools

from utils.config import load_config
from utils.seeding import seed_all
from utils.logger import Logger
from env.connect_n_env import ConnectNEnv
from agents.actor_critic import ActorCritic


def compute_gae(rewards, masks, values, gamma, lam):
    """
    Generalized Advantage Estimation (GAE).
    """
    advantages = []
    gae = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * masks[i] - values[i]
        gae = delta + gamma * lam * masks[i] * gae
        advantages.insert(0, gae)
    return advantages


def update_agents(batch_trajs, agent_A, agent_B, opt_A, opt_B, device, cfg, logger, ep,
                  batch_metrics):
    """
    Update both agents using batched trajectories, log metrics, and collect for plotting.
    """
    for key, agent, optimizer in [
        ("A", agent_A, opt_A),
        ("B", agent_B, opt_B),
    ]:
        ep_trajs = batch_trajs[key]
        # per-episode returns
        per_ep_returns = [sum(tr["rewards"]) for tr in ep_trajs]
        avg_return = float(sum(per_ep_returns) / len(ep_trajs))

        # flatten
        flat_rewards = list(itertools.chain.from_iterable(tr["rewards"] for tr in ep_trajs))
        flat_masks   = list(itertools.chain.from_iterable(tr["masks"]   for tr in ep_trajs))
        flat_values  = list(itertools.chain.from_iterable(tr["values"]  for tr in ep_trajs))
        flat_log_probs = list(itertools.chain.from_iterable(tr["log_probs"] for tr in ep_trajs))
        flat_entropies = list(itertools.chain.from_iterable(tr["entropies"] for tr in ep_trajs))

        # compute advantages & returns
        advs = compute_gae(flat_rewards, flat_masks, flat_values,
                           cfg["gamma"], cfg["gae_lambda"])
        returns = [a + v for a, v in zip(advs, flat_values[:-1])]

        # to tensors
        log_probs = torch.stack(flat_log_probs)
        values    = torch.tensor(flat_values[:-1], device=device)
        returns_t = torch.tensor(returns, device=device)
        advs_t    = torch.tensor(advs, device=device)
        entropies = torch.stack(flat_entropies)

        # losses
        policy_loss  = -(log_probs * advs_t.detach()).mean()
        value_loss   = F.mse_loss(values, returns_t.detach())
        entropy_loss = entropies.mean()
        loss = (
            policy_loss
            + cfg["value_coef"] * value_loss
            - cfg["entropy_coef"] * entropy_loss
        )

        # step
        optimizer.zero_grad()
        loss.backward()
        # grad clipping
        max_norm = cfg.get("max_grad_norm", 0.0)
        if max_norm and max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm)
        optimizer.step()

        # log to TensorBoard
        logger.log_scalar(f"Reward/{key}", avg_return, ep)
        logger.log_scalar(f"Loss/{key}"  , loss.item(),   ep)

        # collect metrics for Matplotlib
        batch_metrics[f"rewards_{key}"].append(avg_return)
        batch_metrics[f"losses_{key}"].append(loss.item())


def main():
    # 1) Load config & seed
    cfg = load_config()
    seed_all(cfg["seed"])

    # 2) Device
    device = torch.device(
        "cuda" if cfg.get("use_cuda", False) and torch.cuda.is_available()
        else "cpu"
    )

    # 3) Env & dims
    env = ConnectNEnv(
        cfg["grid_rows"],
        cfg["grid_cols"],
        cfg["connect_k"],
        cfg["max_steps_per_episode"]
    )
    obs_dim = cfg["grid_rows"] * cfg["grid_cols"]
    act_dim = cfg["grid_cols"]

    # 4) Agents & optimizers
    agent_A = ActorCritic(obs_dim, act_dim, cfg["actor_layers"]).to(device)
    agent_B = ActorCritic(obs_dim, act_dim, cfg["actor_layers"]).to(device)
    opt_A = optim.Adam(agent_A.parameters(), lr=cfg["learning_rate"])
    opt_B = optim.Adam(agent_B.parameters(), lr=cfg["learning_rate"])

    # 5) Logger
    logger = Logger(cfg["results_dir"])

    # 6) Buffers for batching
    batch_trajs = {"A": [], "B": []}
    batch_metrics = {
        "rewards_A": [], "rewards_B": [],
        "losses_A": [],  "losses_B": [],
    }

    # 7) Training loop
    for ep in range(1, cfg["num_episodes"] + 1):
        # reset & randomize starter
        obs = env.reset()
        env.current_player = 1 if random.random() < 0.5 else 2
        state = torch.tensor(obs, dtype=torch.float32,
                             device=device).flatten().unsqueeze(0)

        # per-episode trajectories
        trajectories = {
            "A": {k: [] for k in ["rewards","masks","values","log_probs","entropies"]},
            "B": {k: [] for k in ["rewards","masks","values","log_probs","entropies"]},
        }
        done = False
        while not done:
            key = "A" if env.current_player == 1 else "B"
            agent = agent_A if key == "A" else agent_B

            dist, value = agent(state)
            action = dist.sample()
            next_obs, reward, done, _ = env.step(action.item())

            trajectories[key]["rewards"].append(reward)
            trajectories[key]["masks"].append(0.0 if done else 1.0)
            trajectories[key]["values"].append(value.item())
            trajectories[key]["log_probs"].append(dist.log_prob(action))
            trajectories[key]["entropies"].append(dist.entropy())

            state = torch.tensor(next_obs, dtype=torch.float32,
                                 device=device).flatten().unsqueeze(0)

        # bootstrap values
        for key in trajectories:
            trajectories[key]["values"].append(0.0)

        # append episode to batch buffer
        batch_trajs["A"].append(trajectories["A"])
        batch_trajs["B"].append(trajectories["B"])

        # perform update if batch is ready
        if len(batch_trajs["A"]) == cfg.get("batch_size", 1):
            update_agents(
                batch_trajs, agent_A, agent_B, opt_A, opt_B,
                device, cfg, logger, ep, batch_metrics
            )
            batch_trajs = {"A": [], "B": []}

        # periodic opponent sync
        if ep % cfg.get("opponent_update_interval", 100) == 0:
            agent_B.load_state_dict(agent_A.state_dict())

        # console logging
        if ep % cfg["log_interval"] == 0:
            print(f"Ep {ep}")

        # checkpoint saving
        if ep % cfg["save_interval"] == 0:
            ckpt = {
                "episode": ep,
                "agent_A_state": agent_A.state_dict(),
                "agent_B_state": agent_B.state_dict(),
                "opt_A_state":   opt_A.state_dict(),
                "opt_B_state":   opt_B.state_dict(),
                "config":        cfg,
            }
            logger.save_checkpoint(ckpt, filename=f"checkpoint_{ep}.pt")

    # flush any remaining episodes
    if batch_trajs["A"]:
        update_agents(
            batch_trajs, agent_A, agent_B, opt_A, opt_B,
            device, cfg, logger, ep, batch_metrics
        )

    # final plots
    logger.plot_curve(batch_metrics["rewards_A"], "BatchRewards_A", "batch_rewards_A.png")
    logger.plot_curve(batch_metrics["rewards_B"], "BatchRewards_B", "batch_rewards_B.png")
    logger.plot_curve(batch_metrics["losses_A"],  "BatchLosses_A",  "batch_losses_A.png")
    logger.plot_curve(batch_metrics["losses_B"],  "BatchLosses_B",  "batch_losses_B.png")

    # final checkpoint
    final_ckpt = {
        "episode": ep,
        "agent_A_state": agent_A.state_dict(),
        "agent_B_state": agent_B.state_dict(),
        "opt_A_state":   opt_A.state_dict(),
        "opt_B_state":   opt_B.state_dict(),
        "config":        cfg,
    }
    logger.save_checkpoint(final_ckpt, filename="final_checkpoint.pt")
    print(f"Saved final checkpoint to {logger.log_dir}/final_checkpoint.pt")


if __name__ == "__main__":
    main()
