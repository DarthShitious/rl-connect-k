import random
from env.connect_n_env import ConnectNEnv

def random_agent(env):
    return random.randrange(env.action_space.n)

def evaluate_agent(agent, cfg, num_games=100):
    win_counts = {"agent": 0, "draw": 0, "opp": 0}
    for _ in range(num_games):
        env = ConnectNEnv(cfg)
        state = env.reset()
        done = False
        while not done:
            # Agent move
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            if done:
                if reward > 0:
                    win_counts["agent"] += 1
                elif reward == 0:
                    win_counts["draw"] += 1
                else:
                    win_counts["opp"] += 1
                break
            # Opponent move (random)
            opp_action = random_agent(env)
            state, reward, done, _ = env.step(opp_action)
            if done:
                if reward > 0:
                    win_counts["opp"] += 1
                elif reward == 0:
                    win_counts["draw"] += 1
                else:
                    win_counts["agent"] += 1
                break
    win_rate = win_counts["agent"] / num_games
    return win_rate, win_counts
