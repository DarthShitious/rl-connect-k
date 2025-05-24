# train.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import random

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

def main():
    # 1) Load configuration and seed
    cfg = load_config()
    seed_all(cfg["seed"])

    # 2) Device setup
    device = torch.device(
        "cuda" if cfg.get("use_cuda", False) and torch.cuda.is_available()
        else "cpu"
    )

    # 3) Environment & dimensions
    env = ConnectNEnv(
        cfg["grid_rows"],
        cfg["grid_cols"],
        cfg["connect_k"],
        cfg["max_steps_per_episode"]
    )
    obs_dim = cfg["grid_rows"] * cfg["grid_cols"]
    act_dim = cfg["grid_cols"]

    # 4) Instantiate two agents and optimizers
    agent_A = ActorCritic(obs_dim, act_dim, cfg["actor_layers"]).to(device)
    agent_B = ActorCritic(obs_dim, act_dim, cfg["actor_layers"]).to(device)
    opt_A = optim.Adam(agent_A.parameters(), lr=cfg["learning_rate"])
    opt_B = optim.Adam(agent_B.parameters(), lr=cfg["learning_rate"])

    # 5) Logger for TensorBoard & plots
    logger = Logger(cfg["results_dir"])

    # 6) Storage for metrics
    all_rewards_A, all_rewards_B = [], []
    all_losses_A, all_losses_B = [], []

    # 7) Training loop
    for ep in range(1, cfg["num_episodes"] + 1):
        # --- reset & randomize who starts this episode ---
        obs = env.reset()
        # 50/50 coin flip: A (player 1) or B (player 2) starts
        env.current_player = 1 if random.random() < 0.5 else 2
        state = torch.tensor(obs, dtype=torch.float32, device=device)\
                      .flatten().unsqueeze(0)

        # trajectories for both agents
        trajectories = {
            "A": {"rewards": [], "masks": [], "values": [], "log_probs": [], "entropies": []},
            "B": {"rewards": [], "masks": [], "values": [], "log_probs": [], "entropies": []},
        }

        done = False
        # --- rollout one episode ---
        while not done:
            current_key = "A" if env.current_player == 1 else "B"
            agent = agent_A if current_key == "A" else agent_B

            dist, value = agent(state)
            action = dist.sample()
            next_obs, reward, done, _ = env.step(action.item())

            trajectories[current_key]["rewards"].append(reward)
            trajectories[current_key]["masks"].append(0.0 if done else 1.0)
            trajectories[current_key]["values"].append(value.item())
            trajectories[current_key]["log_probs"].append(dist.log_prob(action))
            trajectories[current_key]["entropies"].append(dist.entropy())

            state = torch.tensor(next_obs, dtype=torch.float32, device=device)\
                         .flatten().unsqueeze(0)

        # append zero for bootstrap value
        for key in trajectories:
            trajectories[key]["values"].append(0.0)

        # --- update both agents ---
        for key, agent, optimizer, all_rewards, all_losses in [
            ("A", agent_A, opt_A, all_rewards_A, all_losses_A),
            ("B", agent_B, opt_B, all_rewards_B, all_losses_B),
        ]:
            traj = trajectories[key]
            advs = compute_gae(
                traj["rewards"],
                traj["masks"],
                traj["values"],
                cfg["gamma"],
                cfg["gae_lambda"],
            )
            returns = [a + v for a, v in zip(advs, traj["values"][:-1])]

            log_probs = torch.stack(traj["log_probs"])
            values    = torch.tensor(traj["values"][:-1], device=device)
            returns   = torch.tensor(returns, device=device)
            advs      = torch.tensor(advs, device=device)
            entropies = torch.stack(traj["entropies"])

            # losses
            policy_loss  = -(log_probs * advs.detach()).mean()
            value_loss   = F.mse_loss(values, returns.detach())
            entropy_loss = entropies.mean()
            loss = (
                policy_loss
                + cfg["value_coef"] * value_loss
                - cfg["entropy_coef"] * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()

            # optional grad-norm clipping
            max_norm = cfg.get("max_grad_norm", 0.0)
            if max_norm and max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm)

            optimizer.step()

            # log metrics
            all_rewards.append(sum(traj["rewards"]))
            all_losses.append(loss.item())
            logger.log_scalar(f"Reward/{key}", all_rewards[-1], ep)
            logger.log_scalar(f"Loss/{key}",   all_losses[-1], ep)

        # periodically sync B ← A
        if ep % cfg.get("opponent_update_interval", 100) == 0:
            agent_B.load_state_dict(agent_A.state_dict())

        # console output
        if ep % cfg["log_interval"] == 0:
            rA, rB = all_rewards_A[-1], all_rewards_B[-1]
            print(f"Ep {ep}\tReward A: {rA:.2f}\tReward B: {rB:.2f}")

        # save checkpoints
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

    # --- after training ends: plot and final save ---
    logger.plot_curve(all_rewards_A, "Rewards_A", "rewards_A.png")
    logger.plot_curve(all_rewards_B, "Rewards_B", "rewards_B.png")
    logger.plot_curve(all_losses_A,  "Losses_A",  "losses_A.png")
    logger.plot_curve(all_losses_B,  "Losses_B",  "losses_B.png")

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
