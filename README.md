# Connect-N RL Pipeline (Two-Agent Self-Play)

## Setup

```bash
git clone <repo_url>; cd connect_n
pip install -r requirements.txt
```

## Configuration

Edit hyperparams in `configs/default.yaml` or override via `--config`.

Key additions:
- `opponent_update_interval`: episodes between copying Agent A → B.
- `eval_episodes`: number of games in evaluation.
- `checkpoint`: path to .pt in your eval YAML.

## Training

```bash
python train.py --config configs/default.yaml
```

This spins up **Agent A** and **Agent B** playing self-play. Checkpoints and logs saved to `results/{timestamp}/`.

## Evaluation

To evaluate **Agent A** (default):
```bash
python evaluate.py --config configs/exp01.yaml
```
where `configs/exp01.yaml` contains:
```yaml
checkpoint: "/path/to/results/.../final_checkpoint.pt"
eval_episodes: 200
```

## Monitoring

- **TensorBoard**:
  ```bash
  tensorboard --logdir results/
  ```
- **Matplotlib** saved plots in each run’s directory.

## Extending

- Swap in different policy nets in `agents/`.
- Add new envs under `env/`.
- Adjust training logic in `train.py`.
