import numpy as np
from typing import Tuple

class ConnectNEnv:
    """
    Simple Connect-N environment.
    State: 2D grid with values {0=empty, 1=player1, 2=player2}
    Actions: column index to drop a piece.
    """
    def __init__(self, rows: int, cols: int, k: int, max_steps: int):
        self.rows = rows
        self.cols = cols
        self.k = k
        self.max_steps = max_steps
        self.reset()

    def reset(self, start_player: int = 1):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = start_player
        self.steps = 0
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Drop a piece in column action.
        Returns (obs, reward, done, info).
        """
        col = action
        for r in reversed(range(self.rows)):
            if self.board[r, col] == 0:
                self.board[r, col] = self.current_player
                break
        else:
            return self._get_obs(), -10.0, True, {"invalid": True}

        self.steps += 1
        win = self._check_win(r, col)
        done = win or self.steps >= self.max_steps or np.all(self.board != 0)
        reward = 1.0 if win else 0.0
        self.current_player = 1 if self.current_player == 2 else 2
        return self._get_obs(), reward, done, {}

    def _get_obs(self) -> np.ndarray:
        return self.board.copy()

    def _check_win(self, r: int, c: int) -> bool:
        """Check if placing at (r,c) made k in a row."""
        b = self.board
        p = b[r, c]
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for dr, dc in directions:
            count = 1
            for s in [1, -1]:
                rr, cc = r + s*dr, c + s*dc
                while 0 <= rr < self.rows and 0 <= cc < self.cols and b[rr, cc] == p:
                    count += 1
                    rr += s*dr
                    cc += s*dc
            if count >= self.k:
                return True
        return False
