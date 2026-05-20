"""
Early Stopping Module

Monitors validation accuracy during training and stops early when
no improvement is observed for a specified number of consecutive epochs.

Author: Claude
Date: 2026-02-13
"""

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early Stopping Class

    Monitors validation accuracy and sets early_stop=True when no
    improvement is observed for `patience` consecutive epochs.

    Args:
        patience (int): Number of epochs to wait. Default 15.
            Triggers early stopping after this many epochs without improvement.
        verbose (bool): Whether to print log messages. Default False.

    Attributes:
        patience (int): Number of epochs to wait
        counter (int): Current count of consecutive epochs without improvement
        best_score (float): Current best validation accuracy
        best_epoch (int): Epoch corresponding to best accuracy
        early_stop (bool): Whether early stopping should be triggered

    Example:
        >>> stopper = EarlyStopping(patience=15)
        >>> for epoch in range(100):
        ...     val_acc = train_and_validate(...)
        ...     if stopper(val_acc, epoch):
        ...         print(f"Early stopping triggered at epoch {epoch}")
        ...         break
        >>> summary = stopper.get_summary()
        >>> print(f"Best accuracy: {summary['best_score']:.4f} (epoch {summary['best_epoch']})")
    """

    def __init__(self, patience=15, verbose=False):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait, default 15 (consistent with all training scripts)
            verbose: Whether to print log messages
        """
        self.patience = patience
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

        if self.verbose:
            logger.info(f"Early stopping initialized: patience={patience}")

    def __call__(self, val_score, epoch):
        """
        Update early stopping state.

        Args:
            val_score: Validation accuracy for current epoch
            epoch: Current epoch number (1-based)

        Returns:
            bool: Whether early stopping should be triggered
        """
        # First call: set baseline
        if self.best_score is None:
            self.best_score = val_score
            self.best_epoch = epoch
            if self.verbose:
                logger.info(f"Epoch {epoch}: Baseline accuracy set to {val_score:.4f}")
            return False

        # Check for improvement (strictly greater counts as improvement, equal does not)
        is_better = val_score > self.best_score

        if is_better:
            # Accuracy improved, reset counter
            if self.verbose and val_score > self.best_score:
                logger.info(f"Epoch {epoch}: Accuracy improved {self.best_score:.4f} -> {val_score:.4f}, counter reset")

            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            # No improvement, increment counter
            self.counter += 1

            if self.verbose:
                logger.info(f"Epoch {epoch}: No improvement ({val_score:.4f} < {self.best_score:.4f}), "
                           f"counter: {self.counter}/{self.patience}")

            # Check if early stopping condition is met
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered! {self.counter} consecutive epochs without improvement")
                logger.info(f"Best accuracy: {self.best_score:.4f} (epoch {self.best_epoch})")
                return True

            return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

        if self.verbose:
            logger.info("Early stopping reset")

    def get_summary(self):
        """
        Get early stopping state summary.

        Returns:
            dict: Dictionary with the following keys
                - best_score: Best accuracy
                - best_epoch: Epoch corresponding to best accuracy
                - current_epoch: Current epoch number (best_epoch + counter)
                - counter: Current count of consecutive epochs without improvement
                - patience: Number of epochs to wait
                - early_stop: Whether early stopping was triggered
                - remaining_epochs: Remaining tolerable epochs without improvement
        """
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'current_epoch': self.best_epoch + self.counter if self.best_epoch is not None else None,
            'counter': self.counter,
            'patience': self.patience,
            'early_stop': self.early_stop,
            'remaining_epochs': max(0, self.patience - self.counter)
        }

    def __repr__(self):
        """String representation."""
        if self.best_score is None:
            return f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, not started)"
        else:
            return (f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, "
                    f"best={self.best_score:.4f}@epoch{self.best_epoch}, "
                    f"counter={self.counter}/{self.patience})")
