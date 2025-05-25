import os
import time
import argparse
from env.connect_n_env import ConnectNEnv
from agents.dqn_agent import DQNAgent
from utils.config import load_config
from utils.seeding import seed_all
from utils.logger import Logger
from evaluate import evaluate_agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    logger = Logger(results_dir)

    seed_all(cfg.random_seed)
    env = ConnectNEnv(cfg)
    agent = DQNAgent(cfg, env)
    batch_size = cfg.batch_size

    for ep in range(1, cfg.num_episodes + 1):
        state = env.reset()
        total_reward = 0
        loss = None
        for t in range(cfg.max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.update(batch_size)
            state = next_state
            total_reward += reward
            if done:
                break
        logger.log("Reward/train", total_reward, ep)
        if loss is not None:
            logger.log("Loss/train", loss, ep)
        if ep % cfg.evaluate_every == 0:
            win_rate, _ = evaluate_agent(agent, cfg, num_games=100)
            logger.log("WinRate/random", win_rate, ep)
        if ep % 1000 == 0:
            agent.save(os.path.join(results_dir, f"checkpoint_{ep}.pt"))
    logger.close()

if __name__ == "__main__":
    main()
