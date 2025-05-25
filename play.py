import argparse
from env.connect_n_env import ConnectNEnv
from agents.dqn_agent import DQNAgent
from utils.config import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = ConnectNEnv(cfg)
    agent = DQNAgent(cfg, env)
    agent.load(args.checkpoint)

    state = env.reset()
    done = False
    while not done:
        env.render()
        action = int(input(f"Select column (0-{env.cols-1}): "))
        state, reward, done, _ = env.step(action)
        if done:
            break
        ai_action = agent.select_action(state)
        print(f"AI plays column {ai_action}")
        state, reward, done, _ = env.step(ai_action)
    env.render()
    if reward > 0:
        print("You win!")
    elif reward == 0:
        print("Draw!")
    else:
        print("AI wins!")

if __name__ == "__main__":
    main()
