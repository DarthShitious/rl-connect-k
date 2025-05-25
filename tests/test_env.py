import pytest
import numpy as np
from types import SimpleNamespace
from env.connect_n_env import ConnectNEnv

@pytest.fixture
def cfg():
    return SimpleNamespace(
        grid_rows=4, grid_cols=4, connect_k=3,
        max_steps_per_episode=16, random_seed=0
    )

def test_reset(cfg):
    env = ConnectNEnv(cfg)
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (cfg.grid_rows, cfg.grid_cols)
    assert np.all(obs == 0)

def test_step_and_win(cfg):
    env = ConnectNEnv(cfg)
    env.reset()
    # horizontal win
    for a in [0, 1, 2]:
        obs, r, done, _ = env.step(a)
    obs, r, done, _ = env.step(3)
    assert done
    assert r == 1
