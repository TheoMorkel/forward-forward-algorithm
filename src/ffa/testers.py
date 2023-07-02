from typing import Any

import torch

import wandb
from src.datasets import Dataset
from src.ffa.models import FFFNN
from src.metrics import accuracy


class Tester:
    def test(self, model: FFFNN, dataset: Dataset, device: Any | str = "cpu"):
        metrics = {}
        # Set model in testing mode
        model.to(device)
        model.eval()

        # Test Evaluation
        test_accuracies = []
        for x, y in dataset.test:
            x, y = x.to(device), y.to(device)
            pred_y = model.predict(x)
            acc = accuracy(pred_y, y)
            test_accuracies.append(acc)
        mean_test_accuracy = torch.mean(torch.tensor(test_accuracies))
        mean_test_err = 1.0 - mean_test_accuracy
        print(
            f"test accuracy: {mean_test_accuracy*100:.2f}%, test error: {(1.0 - mean_test_accuracy)*100:.2f}%"
        )
        metrics['test/error'] = mean_test_err
        metrics['test/step'] = 1
        wandb.log(metrics)
        wandb.run.summary["test_error"] = mean_test_err
        wandb.run.summary["test_accuracy"] = mean_test_accuracy
