import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, results_dir: str):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(results_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb = SummaryWriter(log_dir=self.log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar to TensorBoard."""
        self.tb.add_scalar(tag, value, step)

    def save_checkpoint(self, state: dict, filename: str = "checkpoint.pt"):
        path = os.path.join(self.log_dir, filename)
        torch.save(state, path)

    def plot_curve(self, values: list, title: str, filename: str):
        """Save a Matplotlib plot (e.g. loss curve)."""
        plt.figure()
        plt.plot(values)
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel(title)
        plt.savefig(os.path.join(self.log_dir, filename))
        plt.close()
