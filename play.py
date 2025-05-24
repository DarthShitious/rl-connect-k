#!/usr/bin/env python3
# play.py

import argparse
import yaml
import torch

from utils.seeding import seed_all
from env.connect_n_env import ConnectNEnv
from agents.actor_critic import ActorCritic, BoardTransformer


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
    parser = argparse.ArgumentParser("Play Connect-N vs. a trained agent")
    parser.add_argument("--config",     type=str, required=True,
                        help="Path to YAML config (e.g. configs/default.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a .pt checkpoint (must contain 'agent_A_state')")
    parser.add_argument("--human_player", type=int, choices=[1,2], default=1,
                        help="Which side you play: 1 (X) or 2 (O)")
    args = parser.parse_args()

    # load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # seed & device
    seed_all(cfg.get("seed", 0))
    device = torch.device(
        "cuda" if cfg.get("use_cuda", False) and torch.cuda.is_available()
        else "cpu"
    )

    # setup environment
    env = ConnectNEnv(
        cfg["grid_rows"], cfg["grid_cols"],
        cfg["connect_k"], cfg["max_steps_per_episode"]
    )
    rows, cols = cfg["grid_rows"], cfg["grid_cols"]
    act_dim = cols

    # instantiate model
    model_type = cfg.get("model_type", "mlp")
    if model_type == "transformer":
        tcfg = cfg["transformer"]
        model = BoardTransformer(
            rows, cols, act_dim,
            d_model=tcfg["d_model"], nhead=tcfg["nhead"],
            layers=tcfg["layers"], dropout=tcfg["dropout"]
        ).to(device)
    else:
        obs_dim = rows * cols
        model = ActorCritic(obs_dim, act_dim, cfg["actor_layers"]).to(device)

    # load weights
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["agent_A_state"])
    model.eval()

    human = args.human_player
    ai = 2 if human == 1 else 1
    print(f"You are player {human} ({'X' if human==1 else 'O'}), AI is player {ai}.\n")

    # initialize board
    board = env.reset()
    render_board(board)

    while True:
        mover = env.current_player
        if mover == human:
            # human move
            while True:
                try:
                    col = int(input(f"Your move (0–{cols-1}): "))
                except ValueError:
                    print("↳ enter an integer column index.")
                    continue
                if not (0 <= col < cols):
                    print("↳ out of range, try again.")
                    continue
                if board[0, col] != 0:
                    print("↳ that column is full.")
                    continue
                break
            print(f"→ You drop in column {col}\n")
        else:
            # AI move
            if model_type == "transformer":
                st = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                st = torch.tensor(board, dtype=torch.float32, device=device).flatten().unsqueeze(0)
            with torch.no_grad():
                dist, _ = model(st)
                col = int(dist.probs.argmax(dim=-1).item())
            print(f"→ AI (player {ai}) drops in column {col}\n")

        board, reward, done, _ = env.step(col)
        render_board(board)

        if done:
            if reward > 0:
                print("🎉 You win!" if mover == human else "😞 AI wins!")
            else:
                print("— Draw! —")
            break

if __name__ == "__main__":
    main()
