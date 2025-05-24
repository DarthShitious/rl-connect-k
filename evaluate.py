import torch
from utils.config import load_config
from utils.seeding import seed_all
from env.connect_n_env import ConnectNEnv
from agents.actor_critic import ActorCritic

def evaluate(model, env, device, episodes=100):
    model.eval()
    wins = 0
    with torch.no_grad():
        for _ in range(episodes):
            state = torch.tensor(env.reset(), dtype=torch.float32, device=device).flatten().unsqueeze(0)
            done = False
            while not done:
                dist, _ = model(state)
                action = dist.probs.argmax().unsqueeze(0).to(device)
                next_obs, reward, done, _ = env.step(action.item())
                state = torch.tensor(next_obs, dtype=torch.float32, device=device).flatten().unsqueeze(0)
            wins += int(reward > 0)
    return wins / episodes

def main():
    cfg = load_config()
    seed_all(cfg["seed"])
    device = torch.device("cuda" if cfg.get("use_cuda", False) and torch.cuda.is_available()
                          else "cpu")

    env = ConnectNEnv(
        cfg["grid_rows"],
        cfg["grid_cols"],
        cfg["connect_k"],
        cfg["max_steps_per_episode"]
    )
    obs_dim = cfg["grid_rows"] * cfg["grid_cols"]
    act_dim = cfg["grid_cols"]

    model = ActorCritic(obs_dim, act_dim, cfg["actor_layers"]).to(device)

    checkpoint_path = cfg.get("checkpoint", None)
    if checkpoint_path is None:
        raise ValueError("Please set `checkpoint:` in your config YAML.")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["agent_A_state"])

    episodes = cfg.get("eval_episodes", 100)
    win_rate = evaluate(model, env, device, episodes=episodes)
    print(f"Win rate over {episodes} games: {win_rate*100:.2f}%")

if __name__ == "__main__":
    main()
