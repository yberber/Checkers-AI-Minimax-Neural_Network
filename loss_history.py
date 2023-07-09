import utils


class TrainingLossHistory:
    """Keeps track of the training and validation loss history during model training.

    Attributes:
    history (dict): A dictionary that stores the training and validation losses per epoch.
    """

    def __init__(self):
        """Initializes an instance of the TrainingLossHistory class."""
        self.history = dict()
        self.history["train_loss"] = []
        self.history["train_policy_loss"] = []
        self.history["train_value_loss"] = []
        self.history["val_loss"] = []
        self.history["val_policy_loss"] = []
        self.history["val_value_loss"] = []

    def add_loss(self, train_loss, validation_loss):
        """Adds the average loss, policy loss, and value loss for both training and validation to the history.

        Args:
        train_loss (tuple): A tuple containing the average training loss, policy loss, and value loss.
        validation_loss (tuple): A tuple containing the average validation loss, policy loss, and value loss.
        """
        avg_train_loss, avg_train_policy_loss, avg_train_value_loss = train_loss
        avg_val_loss, avg_val_policy_loss, avg_val_value_loss = validation_loss

        self.history["train_loss"].append(avg_train_loss)
        self.history["train_policy_loss"].append(avg_train_policy_loss)
        self.history["train_value_loss"].append(avg_train_value_loss)
        self.history["val_loss"].append(avg_val_loss)
        self.history["val_policy_loss"].append(avg_val_policy_loss)
        self.history["val_value_loss"].append(avg_val_value_loss)

    def plot_history(self, model_iter):
        """Plots the history of training and validation losses.

        Args:
        model_iteration (int): The model iteration at which the losses are being recorded.

        Returns:
        plot_filename (str): The filename of the saved plot.
        """
        plot_filename = utils.plot_loss_history(self.history, model_iter )
        return plot_filename

