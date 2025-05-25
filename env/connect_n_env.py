import numpy as np
from typing import Tuple, Dict

class ConnectNEnv:
    """
    Connect-N environment with reward shaping:
    - win_reward: reward for winning move
    - lose_reward: reward for loss or draw
    - step_reward: reward for non-terminal valid moves
    - invalid_reward: reward for invalid move
    - shaping_bonus: small bonus for creating (k-1)-in-a-row
    """
    def __init__(
        self,
        rows: int,
        cols: int,
        k: int,
        max_steps: int,
        win_reward: float = 1.0,
        lose_reward: float = -1.0,
        step_reward: float = 0.0,
        invalid_reward: float = -1.0,
        shaping_bonus: float = 0.1
    ):
        self.rows = rows
        self.cols = cols
        self.k = k
        self.max_steps = max_steps
        # reward parameters
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.step_reward = step_reward
        self.invalid_reward = invalid_reward
        self.shaping_bonus = shaping_bonus
        self.reset()

    def reset(self, start_player: int = 1) -> np.ndarray:
        """
        Reset board and starting player.
        """
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = start_player
        self.steps = 0
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply action (drop piece into column). Returns (obs, reward, done, info).
        """
        # invalid move if column full
        if self.board[0, action] != 0:
            return self._get_obs(), self.invalid_reward, True, {"invalid": True}

        # drop piece
        for r in range(self.rows-1, -1, -1):
            if self.board[r, action] == 0:
                self.board[r, action] = self.current_player
                placed_r, placed_c = r, action
                break

        self.steps += 1

        # check win
        win = self._check_win(placed_r, placed_c)
        full = np.all(self.board != 0)
        timeout = self.steps >= self.max_steps
        done = win or full or timeout

        # reward shaping: bonus for creating (k-1) in a row
        bonus = 0.0
        b = self.board
        p = b[placed_r, placed_c]
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                rr, cc = placed_r + sign*dr, placed_c + sign*dc
                while 0 <= rr < self.rows and 0 <= cc < self.cols and b[rr, cc] == p:
                    count += 1
                    rr += sign*dr
                    cc += sign*dc
            if count == self.k - 1:
                bonus += self.shaping_bonus

        # assign base reward
        if win:
            reward = self.win_reward + bonus
        elif full or timeout:
            reward = self.lose_reward + bonus
        else:
            reward = self.step_reward + bonus

        # switch player
        self.current_player = 1 if self.current_player == 2 else 2
        return self._get_obs(), reward, done, {}

    def _get_obs(self) -> np.ndarray:
        """
        Return a copy of the board state.
        """
        return self.board.copy()

    def _check_win(self, r: int, c: int) -> bool:
        """
        Check if the last move at (r,c) completed k in a row.
        """
        b = self.board
        p = b[r, c]
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                rr, cc = r + sign*dr, c + sign*dc
                while 0 <= rr < self.rows and 0 <= cc < self.cols and b[rr, cc] == p:
                    count += 1
                    rr += sign*dr
                    cc += sign*dc
            if count >= self.k:
                return True
        return False
