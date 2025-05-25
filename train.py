
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
from torch.distributions import Categorical


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


def update_agents(batch_trajs, model_A, model_B,
                  opt_A, opt_B, device, cfg, logger, ep, batch_metrics):
    """
    Update both agents using batched trajectories, log metrics.
    Curriculum Phase 1: B is a fixed random opponent (frozen) for the first cfg["random_start_episodes"].
    """
    phase1 = ep <= cfg.get("random_start_episodes", 0)

    for key, model, optim in [
        ("A", model_A, opt_A),
        ("B", model_B, opt_B),
    ]:
        # Determine if this agent should be frozen in Phase 1
        freeze = (key == "B" and phase1)

        ep_trajs = batch_trajs[key]
        avg_return = sum(sum(tr["rewards"]) for tr in ep_trajs) / len(ep_trajs)

        # flatten
        flat_rewards   = list(itertools.chain.from_iterable(tr["rewards"]    for tr in ep_trajs))
        flat_masks     = list(itertools.chain.from_iterable(tr["masks"]      for tr in ep_trajs))
        flat_log_probs = list(itertools.chain.from_iterable(tr["log_probs"]  for tr in ep_trajs))
        flat_entropies = list(itertools.chain.from_iterable(tr["entropies"]  for tr in ep_trajs))
        flat_values    = list(itertools.chain.from_iterable(tr["values"][:-1] for tr in ep_trajs))
        flat_values.append(0.0)

        # advantages & returns
        advs = compute_gae(flat_rewards, flat_masks, flat_values,
                           cfg["gamma"], cfg["gae_lambda"])
        returns = [a + v for a, v in zip(advs, flat_values[:-1])]

        # to tensors
        log_probs = torch.stack(flat_log_probs)
        values    = torch.tensor(flat_values[:-1], device=device)
        returns_t = torch.tensor(returns,     device=device)
        advs_t    = torch.tensor(advs,        device=device)
        entropies = torch.stack(flat_entropies)

        # normalize advantages
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)
        advs_t = advs_t * cfg.get("policy_coef", 1.0)

        # compute losses
        policy_loss  = -(log_probs * advs_t.detach()).mean()
        value_loss   = F.mse_loss(values, returns_t.detach())
        entropy_loss = entropies.mean()
        loss = (
              policy_loss
            + cfg["value_coef"] * value_loss
            - cfg["entropy_coef"] * entropy_loss
        )

        # backward
        optim.zero_grad()
        loss.backward()
        # gradient clipping
        max_norm = cfg.get("max_grad_norm", 0.0)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # step optimizer only if not frozen
        if not freeze:
            optim.step()

        # logging
        logger.log_scalar(f"Reward/{key}"    , avg_return       , ep)
        logger.log_scalar(f"PolicyLoss/{key}" , policy_loss.item(), ep)
        logger.log_scalar(f"ValueLoss/{key}"  , value_loss.item() , ep)
        logger.log_scalar(f"Entropy/{key}"    , entropy_loss.item(), ep)
        logger.log_scalar(f"Loss/{key}"       , loss.item()       , ep)

        # update batch metrics
        batch_metrics[f"rewards_{key}"].append(avg_return)
        batch_metrics[f"policy_losses_{key}"].append(policy_loss.item())
        batch_metrics[f"value_losses_{key}"].append(value_loss.item())
        batch_metrics[f"entropies_{key}"].append(entropy_loss.item())
        batch_metrics[f"losses_{key}"].append(loss.item())

    # end of update_agents

def main():
    cfg = load_config()
    seed_all(cfg["seed"])
    device = torch.device(
        "cuda" if cfg.get("use_cuda", False) and torch.cuda.is_available()
        else "cpu"
    )

    # epsilon schedule
    eps_start = cfg.get("epsilon_start", 0.3)
    eps_end   = cfg.get("epsilon_end",   0.05)
    eps_decay = cfg.get("epsilon_decay_episodes", cfg.get("num_episodes", 1))
    def get_epsilon(ep):
        frac = min(ep / eps_decay, 1.0)
        return eps_start + frac * (eps_end - eps_start)

    # state helper
    def make_state(obs_np):
        t = torch.tensor(obs_np, dtype=torch.float32, device=device)
        if cfg.get("model_type", "mlp") == "transformer":
            return t.unsqueeze(0)
        else:
            return t.flatten().unsqueeze(0)

    # environment
    env = ConnectNEnv(
        cfg["grid_rows"], cfg["grid_cols"],
        cfg["connect_k"], cfg["max_steps_per_episode"]
    )
    rows, cols = cfg["grid_rows"], cfg["grid_cols"]
    act_dim = cols

    # instantiate models
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

    # separate optimizers with LR groups
    actor_lr  = cfg.get("actor_lr", cfg.get("learning_rate", 3e-4))
    critic_lr = cfg.get("critic_lr", cfg.get("learning_rate", 1e-4))
    opt_A = optim.Adam([
        {"params": model_A.policy_head.parameters(), "lr": actor_lr},
        {"params": model_A.shared.parameters(),      "lr": critic_lr},
        {"params": model_A.value_head.parameters(),  "lr": critic_lr},
    ])
    opt_B = optim.Adam([
        {"params": model_B.policy_head.parameters(), "lr": actor_lr},
        {"params": model_B.shared.parameters(),      "lr": critic_lr},
        {"params": model_B.value_head.parameters(),  "lr": critic_lr},
    ])

    # prepare logging and buffers
    logger = Logger(cfg["results_dir"])
    batch_trajs   = {"A": [], "B": []}
    batch_metrics = {
        "rewards_A": [], "rewards_B": [],
        "policy_losses_A": [], "policy_losses_B": [],
        "value_losses_A": [],  "value_losses_B": [],
        "entropies_A": [],      "entropies_B": [],
        "losses_A": [],         "losses_B": []
    }
    random_start = cfg.get("random_start_episodes", 0)

    # training loop
    for ep in range(1, cfg["num_episodes"] + 1):
        # flip a coin to decide who starts
        start_player = 1 if random.random() < 0.5 else 2
        obs = env.reset(start_player=start_player)
        env.current_player = 1 if random.random() < 0.5 else 2
        state = make_state(obs)
        epsilon = get_epsilon(ep)

        trajectories = {
            "A": {k: [] for k in ["rewards","masks","values","log_probs","entropies"]},
            "B": {k: [] for k in ["rewards","masks","values","log_probs","entropies"]},
        }
        done = False
        while not done:
            key = "A" if env.current_player == 1 else "B"
            model = model_A if key == "A" else model_B

            dist, value = model(state)
            valid_mask = torch.from_numpy((obs[0] == 0)).to(device).unsqueeze(0)
            logits = dist.logits.masked_fill(~valid_mask, float("-1e9"))
            dist   = Categorical(logits=logits)

            # action selection
            if key == "B" and ep <= random_start:
                valid = [c for c in range(cols) if obs[0, c] == 0]
                action = torch.tensor(random.choice(valid), device=device)
            else:
                if random.random() < epsilon:
                    valid = [c for c in range(cols) if obs[0, c] == 0]
                    action = torch.tensor(random.choice(valid), device=device)
                else:
                    action = dist.sample()

            next_obs, reward, done, _ = env.step(action.item())
            traj = trajectories[key]
            traj["rewards"].append(reward)
            traj["masks"].append(0.0 if done else 1.0)
            traj["values"].append(value.item())
            traj["log_probs"].append(dist.log_prob(action))
            traj["entropies"].append(dist.entropy())

            obs   = next_obs
            state = make_state(obs)

        # bootstrap and append
        for k in trajectories:
            trajectories[k]["values"].append(0.0)
        batch_trajs["A"].append(trajectories["A"])
        batch_trajs["B"].append(trajectories["B"])

        # update
        if len(batch_trajs["A"]) == cfg.get("batch_size", 1):
            update_agents(batch_trajs, model_A, model_B,
                          opt_A, opt_B, device, cfg, logger, ep, batch_metrics)
            batch_trajs = {"A": [], "B": []}

        if ep % cfg.get("log_interval", 10) == 0:
            print(f"Ep {ep}")
        if ep % cfg.get("save_interval", 1000) == 0:
            ckpt = {
                "episode": ep,
                "agent_A_state": model_A.state_dict(),
                "agent_B_state": model_B.state_dict(),
                "opt_A_state":   opt_A.state_dict(),
                "opt_B_state":   opt_B.state_dict(),
                "config":        cfg,
            }
            logger.save_checkpoint(ckpt, filename=f"checkpoint_{ep}.pt")

    # final update & plots
    if batch_trajs["A"]:
        update_agents(batch_trajs, model_A, model_B,
                      opt_A, opt_B, device, cfg, logger, ep, batch_metrics)

    # plot metrics
    for metric, fname in [
        ("rewards_A",      "batch_rewards_A.png"),
        ("rewards_B",      "batch_rewards_B.png"),
        ("policy_losses_A","batch_policy_loss_A.png"),
        ("policy_losses_B","batch_policy_loss_B.png"),
        ("value_losses_A", "batch_value_loss_A.png"),
        ("value_losses_B", "batch_value_loss_B.png"),
        ("entropies_A",    "batch_entropy_A.png"),
        ("entropies_B",    "batch_entropy_B.png"),
        ("losses_A",       "batch_losses_A.png"),
        ("losses_B",       "batch_losses_B.png"),
    ]:
        agent, metric_name = metric.split("_")[0], metric
        if batch_metrics[metric_name]:
            logger.plot_curve(batch_metrics[metric_name],
                              metric_name, fname)

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

