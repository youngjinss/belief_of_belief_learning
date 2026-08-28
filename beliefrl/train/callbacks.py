"""Training callbacks."""

import torch

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in validation loss to qualify as improvement
            restore_best_weights: Whether to restore model weights from the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        """
        Call this method after each epoch

        Args:
            val_loss: Current validation loss
            model: Model to potentially store weights from

        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                # Handle DataParallel models
                if isinstance(model, torch.nn.DataParallel):
                    self.best_weights = {
                        k: v.clone() for k, v in model.module.state_dict().items()
                    }
                else:
                    self.best_weights = {
                        k: v.clone() for k, v in model.state_dict().items()
                    }
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                # Handle DataParallel models
                if isinstance(model, torch.nn.DataParallel):
                    model.module.load_state_dict(self.best_weights)
                else:
                    model.load_state_dict(self.best_weights)
            return True
        return False
