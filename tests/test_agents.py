import pytest
import numpy as np
from types import SimpleNamespace
from env.connect_n_env import ConnectNEnv
from agents.dqn_agent import DQNAgent

@pytest.fixture
def cfg():
    return SimpleNamespace(
        grid_rows=4, grid_cols=4, connect_k=3,
        max_steps_per_episode=16, random_seed=0,
        batch_size=2,
        dqn=SimpleNamespace(
            lr=0.001, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.1,
            epsilon_decay=100, target_update_freq=10
        )
    )

def test_dqn_initializes(cfg):
    env = ConnectNEnv(cfg)
    agent = DQNAgent(cfg, env)
    assert agent.policy_net is not None

def test_dqn_select_action(cfg):
    env = ConnectNEnv(cfg)
    agent = DQNAgent(cfg, env)
    state = env.reset()
    action = agent.select_action(state)
    assert 0 <= action < env.action_space.n
