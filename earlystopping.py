import numpy as np
import torch

import utils


class EarlyStopping:
    """
    Early stopping utility.

    This class helps to stop the training when the validation loss does not improve after
    a given number of epochs, referred to as 'patience'. It also supports learning rate
    reduction when improvement plateaus, saving state of the best model, and optionally
    saving multiple models for testing.

    Attributes:
    -----------
    patience : int, default=3
        Number of epochs to wait before stopping the training when the loss is not improving.
    verbose : bool, default=False
        If True, prints a message for each validation loss improvement.
    threshold : float, default=0
        Minimum change in the monitored quantity to qualify as an improvement.
    min_learning_rate : float, default=0.0002
        Minimum learning rate below which the learning rate will not be reduced.
    reduce_learning_rate : bool, default=True
        If True, reduces the learning rate when the improvement plateaus.
    save_multiple_models : bool, default=True
        If True, saves the state of all models, not just the best one.
    """
    def __init__(self, patience=3, verbose=False, delta=0, min_lr=0.0002, reduce_lr=True, save_multiple_models=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.min_lr = min_lr
        self.reduce_lr = reduce_lr
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.epoch_id = 0
        self.save_multiple_models = save_multiple_models


    def __call__(self, val_loss, model, optimizer):
        """
        Evaluate if the training should be stopped or the learning rate reduced.

        Parameters:
        -----------
        validation_loss : float
            Validation loss for this epoch.
        model : PyTorch model
            The model being trained.
        optimizer : PyTorch optimizer
            The optimizer being used for training.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model, optimizer)


        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                if self.reduce_lr and optimizer.param_groups[0]['lr'] > self.min_lr:
                    model.load_state_dict(self.model_state_dict)
                    optimizer.load_state_dict(self.optimizer_state_dict)
                    print("Reducing learning rate")
                    for g in optimizer.param_groups:
                        g['lr'] /= 10
                    self.counter = 0
                else:
                    self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

        self.epoch_id += 1


    def _save_checkpoint(self, val_loss, model, optimizer):
        """
        Save the current model and optimizer states as checkpoint.

        Parameters:
        -----------
        validation_loss : float
            Validation loss for this epoch.
        model : PyTorch model
            The model being trained.
        optimizer : PyTorch optimizer
            The optimizer being used for training.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        if self.save_multiple_models:
            utils.save_model_and_optimizer(model, optimizer, self.epoch_id, True, "data/models_to_test/")
        self.val_loss_min = val_loss
        self.model_state_dict = model.state_dict()
        self.optimizer_state_dict = optimizer.state_dict()
