import gym
from gym import spaces
import numpy as np

class ConnectNEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        self.rows = config.grid_rows
        self.cols = config.grid_cols
        self.k = config.connect_k
        self.max_steps = config.max_steps_per_episode
        self.action_space = spaces.Discrete(self.cols)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.rows, self.cols), dtype=np.int8)
        self.seed(config.random_seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        self.heights = [0] * self.cols
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return self.board.copy()

    def step(self, action):
        if self.heights[action] >= self.rows:
            return self._get_obs(), -10, True, {}
        row = self.heights[action]
        self.board[row, action] = self.current_player
        self.heights[action] += 1
        self.steps += 1

        done = False
        reward = 0

        if self._check_win(row, action):
            done = True
            reward = 1
        elif self.steps >= self.max_steps:
            done = True
            reward = 0
        else:
            self.current_player = 2 if self.current_player == 1 else 1

        return self._get_obs(), reward, done, {}

    def _check_win(self, row, col):
        board = self.board
        player = board[row, col]
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            for dir in [1, -1]:
                r, c = row, col
                while True:
                    r += dr * dir
                    c += dc * dir
                    if (0 <= r < self.rows and 0 <= c < self.cols and
                       board[r, c] == player):
                        count += 1
                    else:
                        break
            if count >= self.k:
                return True
        return False

    def render(self, mode='human'):
        print(self.board[::-1])
