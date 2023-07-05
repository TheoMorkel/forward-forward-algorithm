from typing import Any, Callable

import torch
from tqdm import tqdm

import wandb
from src.datasets import Dataset
from src.ffa.models import FFFNN
from src.metrics import accuracy



class GreedyTrainer:
    def __init__(self, epochs: int = 10, save_to: str = "models", evaluate: bool = False):
        self.epochs = epochs
        self.save_to = save_to
        self.evaluate = evaluate

    def __str__(self) -> str:
        return "GreedyTrainer"

    def setEpochs(self, epochs: int):
        self.epochs = epochs
    
    def setSaveTo(self, save_to: str):
        self.save_to = save_to
    
    def setEvaluate(self, evaluate: bool):
        self.evaluate = evaluate
    

    def train(self, model: FFFNN, dataset: Dataset, device: Any | str = "cpu"):
        metrics = {}
        # Set model in training mode
        model.to(device)
        model.train()

        # Main training loop
        for train_layer, _ in enumerate(model.layers):
            for epoch in range(self.epochs):
                pbar = tqdm(dataset.train)
                for b, (x, y) in enumerate(pbar):
                    x, y = x.to(device), y.to(device)

                    # Generate positive and negative samples
                    x_pos = model.embed_label(x, y)
                    rnd = torch.randperm(y.size(0))
                    x_neg = model.embed_label(x, y[rnd])

                    h_pos = x_pos
                    h_neg = x_neg

                    for forward_layer, layer in enumerate(model.layers):
                        if forward_layer < train_layer:
                            with torch.no_grad():
                                h_pos = layer.forward(h_pos)
                                h_neg = layer.forward(h_neg)
                        elif forward_layer == train_layer:
                            (h_pos, g_pos), (h_neg, g_neg), loss = layer.train_layer(
                                h_pos, h_neg
                            )
                            pbar.set_description(
                                f"layer: {train_layer+1}/{len(model.layers)}, epoch: {epoch+1}/{self.epochs}, loss: {loss:.5f}"
                            )
                            metrics['train/loss'] = loss
                            metrics['train/step'] = train_layer * self.epochs * len(dataset.train) + epoch * len(dataset.train) + b
                            metrics['train/epoch'] = epoch
                            metrics['train/layer'] = train_layer

                            metrics[f'train_layer{train_layer}/loss'] = loss
                            metrics[f'train_layer{train_layer}/epoch'] = epoch
                            metrics[f'train_layer{train_layer}/step'] = epoch * len(dataset.train) + b
                            wandb.log(metrics)

        # Train Evaluation
        train_accuracies = []
        for x, y in dataset.train:
            x, y = x.to(device), y.to(device)
            pred_y = model.predict(x)
            acc = accuracy(pred_y, y)
            train_accuracies.append(acc)
        mean_train_accuracy = torch.mean(torch.tensor(train_accuracies))
        mean_train_err = 1.0 - mean_train_accuracy
        print(
            f"train accuracy: {mean_train_accuracy*100:.2f}%, train error: {(1.0 - mean_train_accuracy)*100:.2f}%"
        )
        metrics['train/error'] = mean_train_err
        metrics['train/error_step'] = 1
        wandb.log(metrics)

        wandb.run.summary["train_error"] = mean_train_err
        wandb.run.summary["train_accuracy"] = mean_train_accuracy

        # Validation Evaluation
        if self.evaluate:
            model.eval()

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
            wandb.run.summary["test_error"] = mean_test_err
            wandb.run.summary["test_accuracy"] = mean_test_accuracy


class NonGreedyTrainer:
    def __init__(self, epochs: int = 10, save_to: str = "models", evaluate: bool = False):
        self.epochs = epochs
        self.save_to = save_to
        self.evaluate = evaluate

    def __str__(self) -> str:
        return "NonGreedyTrainer"

    def setEpochs(self, epochs: int):
        self.epochs = epochs
    
    def setSaveTo(self, save_to: str):
        self.save_to = save_to
    
    def setEvaluate(self, evaluate: bool):
        self.evaluate = evaluate
    

    def train(self, model: FFFNN, dataset: Dataset, device: Any | str = "cpu"):

        metrics = {}
        # Set model in training mode
        model.to(device)
        model.train()

        # Main training loop
        for epoch in range(self.epochs):
            pbar = tqdm(dataset.train)
            for b, (x, y) in enumerate(pbar):
                x, y = x.to(device), y.to(device)

                # Generate positive and negative samples
                x_pos = model.embed_label(x, y)
                rnd = torch.randperm(y.size(0))
                x_neg = model.embed_label(x, y[rnd])

                h_pos = x_pos
                h_neg = x_neg

                for train_layer, layer in enumerate(model.layers):
                    # 'detach' the graph after each layer
                    h_pos, h_neg = h_pos.detach(), h_neg.detach()
                    h_pos.requires_grad = True
                    h_neg.requires_grad = True

                    (h_pos, g_pos), (h_neg, g_neg), loss = layer.train(h_pos, h_neg)
                    pbar.set_description(
                        f"layer: {train_layer+1}/{len(model.layers)}, epoch: {epoch+1}/{self.epochs}, loss: {loss:.5f}"
                    )
                    metrics['train/loss'] = loss
                    metrics['train/step'] = train_layer * self.epochs * len(dataset.train) + epoch * len(dataset.train) + b
                    metrics['train/epoch'] = epoch
                    metrics['train/layer'] = train_layer

                    metrics[f'train_layer{train_layer}/loss'] = loss
                    metrics[f'train_layer{train_layer}/epoch'] = epoch
                    metrics[f'train_layer{train_layer}/step'] = epoch * len(dataset.train) + b
                    wandb.log(metrics)

        # Train Evaluation
        train_accuracies = []
        for x, y in dataset.train:
            x, y = x.to(device), y.to(device)
            pred_y = model.predict(x)
            acc = accuracy(pred_y, y)
            train_accuracies.append(acc)
        mean_train_accuracy = torch.mean(torch.tensor(train_accuracies))
        mean_train_err = 1.0 - mean_train_accuracy
        print(
            f"train accuracy: {mean_train_accuracy*100:.2f}%, train error: {(1.0 - mean_train_accuracy)*100:.2f}%"
        )
        metrics['train/error'] = mean_test_err
        metrics['train/error_step'] = 1
        wandb.log(metrics)

        wandb.run.summary["train_error"] = mean_train_err
        wandb.run.summary["train_accuracy"] = mean_train_accuracy

        # Validation Evaluation
        if self.evaluate:
            model.eval()

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
            wandb.run.summary["test_error"] = mean_test_err
            wandb.run.summary["test_accuracy"] = mean_test_accuracy

