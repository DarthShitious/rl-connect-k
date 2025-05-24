#!/usr/bin/env python3
# play.py

import argparse
import yaml
import torch

from utils.seeding import seed_all
from env.connect_n_env import ConnectNEnv
from agents.actor_critic import ActorCritic

def render_board(board):
    """
    Print the Connect-N board to the console.
    0 → '.', 1 → 'X', 2 → 'O'
    """
    rows, cols = board.shape
    symbols = {0: '.', 1: 'X', 2: 'O'}
    for r in range(rows):
        row_str = " ".join(symbols[int(cell)] for cell in board[r])
        print(f"| {row_str} |")
    print("  " + " ".join(str(c) for c in range(cols)))
    print()

def main():
    p = argparse.ArgumentParser("Play Connect-N vs. a trained agent")
    p.add_argument("--config",     type=str, required=True,
                   help="Path to YAML config (e.g. configs/default.yaml)")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a .pt checkpoint (must contain 'agent_A_state')")
    p.add_argument("--human_player", type=int, choices=[1,2], default=1,
                   help="Which side you play: 1 (X) or 2 (O)")
    args = p.parse_args()

    # 1) Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 2) Seed & device
    seed_all(cfg.get("seed", 0))
    device = torch.device("cuda" if cfg.get("use_cuda", False) and torch.cuda.is_available()
                          else "cpu")

    # 3) Build env & model
    env = ConnectNEnv(
        cfg["grid_rows"],
        cfg["grid_cols"],
        cfg["connect_k"],
        cfg["max_steps_per_episode"],
    )
    obs_dim = cfg["grid_rows"] * cfg["grid_cols"]
    act_dim = cfg["grid_cols"]

    model = ActorCritic(obs_dim, act_dim, cfg["actor_layers"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["agent_A_state"])
    model.eval()

    human = args.human_player
    ai = 2 if human == 1 else 1
    print(f"You are player {human} ({'X' if human==1 else 'O'}), AI is player {ai}.\n")

    # 4) Game loop
    board = env.reset()
    render_board(board)

    while True:
        mover = env.current_player
        # — human turn —
        if mover == human:
            while True:
                try:
                    col = int(input(f"Your move (0–{cfg['grid_cols']-1}): "))
                except ValueError:
                    print("↳ enter an integer column index.")
                    continue
                if not (0 <= col < cfg["grid_cols"]):
                    print("↳ out of range, try again.")
                    continue
                if board[0, col] != 0:
                    print("↳ that column is full.")
                    continue
                break
            print(f"→ You drop in column {col}\n")

        # — AI turn —
        else:
            # prepare state for model
            st = torch.tensor(board, dtype=torch.float32, device=device)
            st = st.flatten().unsqueeze(0)
            with torch.no_grad():
                dist, _ = model(st)
                col = int(dist.probs.argmax(dim=-1).item())
            print(f"→ AI (player {ai}) drops in column {col}\n")

        # 5) Step environment
        board, reward, done, info = env.step(col)
        render_board(board)

        # 6) Check for end
        if done:
            mover_last = mover
            if reward > 0:
                if mover_last == human:
                    print("🎉 You win!")
                else:
                    print("😞 AI wins!")
            else:
                print("— Draw! —")
            break

if __name__ == "__main__":
    main()
