# Connect-k Refactored

This is a refactored Connect-k (generalized Connect-4) project with a DQN agent.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --config configs/default.yaml
```

## Evaluation

```bash
python evaluate.py --config configs/default.yaml
```

## Play

```bash
python play.py --config configs/default.yaml --checkpoint results/<timestamp>/checkpoint_10000.pt
```

Trained results and logs are saved under `results/{timestamp}`.
