
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
from agents.actor_critic import ActorCritic, BoardTransformer

# Uncomment to debug in-place ops
# torch.autograd.set_detect_anomaly(True)


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


def update_agents(batch_trajs, model_A, model_B, opt_A, opt_B,
                  device, cfg, logger, ep, batch_metrics):
    """
    Update both agents using batched trajectories, log metrics.
    """
    for key, model, optimizer in [
        ("A", model_A, opt_A),
        ("B", model_B, opt_B),
    ]:
        ep_trajs = batch_trajs[key]
        # average return over episodes in batch
        avg_return = float(
            sum(sum(tr["rewards"]) for tr in ep_trajs) / len(ep_trajs)
        )

        # flatten lists across episodes
        flat_rewards   = list(itertools.chain.from_iterable(tr["rewards"]    for tr in ep_trajs))
        flat_masks     = list(itertools.chain.from_iterable(tr["masks"]      for tr in ep_trajs))
        flat_log_probs = list(itertools.chain.from_iterable(tr["log_probs"]  for tr in ep_trajs))
        flat_entropies = list(itertools.chain.from_iterable(tr["entropies"]  for tr in ep_trajs))
        flat_values    = list(itertools.chain.from_iterable(tr["values"][:-1] for tr in ep_trajs))
        flat_values.append(0.0)

        # compute advantages and returns
        advs = compute_gae(flat_rewards, flat_masks, flat_values,
                           cfg["gamma"], cfg["gae_lambda"])
        returns = [a + v for a, v in zip(advs, flat_values[:-1])]

        # to tensors
        log_probs = torch.stack(flat_log_probs)
        values    = torch.tensor(flat_values[:-1], device=device)
        returns_t = torch.tensor(returns,     device=device)
        advs_t    = torch.tensor(advs,        device=device)
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

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        max_norm = cfg.get("max_grad_norm", 0.0)
        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # log
        logger.log_scalar(f"Reward/{key}", avg_return, ep)
        logger.log_scalar(f"Loss/{key}"  , loss.item(),   ep)
        batch_metrics[f"rewards_{key}"].append(avg_return)
        batch_metrics[f"losses_{key}"].append(loss.item())


def main():
    cfg = load_config()
    seed_all(cfg["seed"])
    device = torch.device(
        "cuda" if cfg.get("use_cuda", False) and torch.cuda.is_available()
        else "cpu"
    )

    # environment
    env = ConnectNEnv(
        cfg["grid_rows"], cfg["grid_cols"],
        cfg["connect_k"], cfg["max_steps_per_episode"]
    )
    rows, cols = cfg["grid_rows"], cfg["grid_cols"]
    act_dim = cols

    # select model class
    if cfg.get("model_type", "mlp") == "transformer":
        tcfg = cfg["transformer"]
        model_A = BoardTransformer(rows, cols, act_dim,
                                    d_model=tcfg["d_model"],
                                    nhead=tcfg["nhead"],
                                    layers=tcfg["layers"],
                                    dropout=tcfg["dropout"]).to(device)
        model_B = BoardTransformer(rows, cols, act_dim,
                                    d_model=tcfg["d_model"],
                                    nhead=tcfg["nhead"],
                                    layers=tcfg["layers"],
                                    dropout=tcfg["dropout"]).to(device)
    else:
        obs_dim = rows * cols
        model_A = ActorCritic(obs_dim, act_dim, cfg["actor_layers"]).to(device)
        model_B = ActorCritic(obs_dim, act_dim, cfg["actor_layers"]).to(device)

    # optimizers & logger
    opt_A = optim.Adam(model_A.parameters(), lr=cfg["learning_rate"])
    opt_B = optim.Adam(model_B.parameters(), lr=cfg["learning_rate"])
    logger = Logger(cfg["results_dir"])

    # batching buffers
    batch_trajs = {"A": [], "B": []}
    batch_metrics = {"rewards_A": [], "rewards_B": [],
                     "losses_A": [],  "losses_B": []}

    for ep in range(1, cfg["num_episodes"] + 1):
        # reset & random start
        obs = env.reset()
        env.current_player = 1 if random.random() < 0.5 else 2

        # build initial state tensor
        if cfg.get("model_type", "mlp") == "transformer":
            state = torch.tensor(obs, dtype=torch.float32,
                                 device=device).unsqueeze(0)  # (1, rows, cols)
        else:
            state = torch.tensor(obs, dtype=torch.float32,
                                 device=device).flatten().unsqueeze(0)

        trajectories = {
            "A": {k: [] for k in ["rewards","masks","values","log_probs","entropies"]},
            "B": {k: [] for k in ["rewards","masks","values","log_probs","entropies"]},
        }
        done = False
        while not done:
            key = "A" if env.current_player == 1 else "B"
            model = model_A if key == "A" else model_B

            dist, value = model(state)
            action = dist.sample()
            next_obs, reward, done, _ = env.step(action.item())

            traj = trajectories[key]
            traj["rewards"].append(reward)
            traj["masks"].append(0.0 if done else 1.0)
            traj["values"].append(value.item())
            traj["log_probs"].append(dist.log_prob(action))
            traj["entropies"].append(dist.entropy())

            # next state
            if cfg.get("model_type", "mlp") == "transformer":
                state = torch.tensor(next_obs, dtype=torch.float32,
                                     device=device).unsqueeze(0)
            else:
                state = torch.tensor(next_obs, dtype=torch.float32,
                                     device=device).flatten().unsqueeze(0)

        # bootstrap values
        for key in trajectories:
            trajectories[key]["values"].append(0.0)

        # add to batch
        batch_trajs["A"].append(trajectories["A"])
        batch_trajs["B"].append(trajectories["B"])

        # update when full
        if len(batch_trajs["A"]) == cfg.get("batch_size", 1):
            update_agents(batch_trajs, model_A, model_B, opt_A, opt_B,
                          device, cfg, logger, ep, batch_metrics)
            batch_trajs = {"A": [], "B": []}

        if ep % cfg.get("log_interval", 10) == 0:
            print(f"Ep {ep}")
        if ep % cfg.get("save_interval", 100) == 0:
            ckpt = {
                "episode": ep,
                "agent_A_state": model_A.state_dict(),
                "agent_B_state": model_B.state_dict(),
                "opt_A_state":   opt_A.state_dict(),
                "opt_B_state":   opt_B.state_dict(),
                "config":        cfg,
            }
            logger.save_checkpoint(ckpt, filename=f"checkpoint_{ep}.pt")

    # final flush
    if batch_trajs["A"]:
        update_agents(batch_trajs, model_A, model_B,
                      opt_A, opt_B, device, cfg, logger, ep, batch_metrics)

    # plot metrics
    logger.plot_curve(batch_metrics["rewards_A"], "BatchRewards_A", "batch_rewards_A.png")
    logger.plot_curve(batch_metrics["rewards_B"], "BatchRewards_B", "batch_rewards_B.png")
    logger.plot_curve(batch_metrics["losses_A"],  "BatchLosses_A",  "batch_losses_A.png")
    logger.plot_curve(batch_metrics["losses_B"],  "BatchLosses_B",  "batch_losses_B.png")

    # final checkpoint
    final_ckpt = {
        "episode": ep,
        "agent_A_state": model_A.state_dict(),
        "agent_B_state": model_B.state_dict(),
        "opt_A_state":   opt_A.state_dict(),
        "opt_B_state":   opt_B.state_dict(),
        "config":        cfg,
    }
    logger.save_checkpoint(final_ckpt, filename="final_checkpoint.pt")
    print(f"Saved final checkpoint to {logger.log_dir}/final_checkpoint.pt")


if __name__ == "__main__":
    main()
