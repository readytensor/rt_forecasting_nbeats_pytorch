import os
import warnings
import math

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from typing import Union
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
from prediction.nbeats_model import NBeatsNet
from sklearn.exceptions import NotFittedError


# Check for GPU availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("device used: ", device)

PREDICTOR_FILE_NAME = "predictor.joblib"
MODEL_PARAMS_FNAME = "model_params.save"
MODEL_WTS_FNAME = "model_wts.save"
HISTORY_FNAME = "history.json"
COST_THRESHOLD = float("inf")


def get_patience_factor(N):
    # magic number - just picked through trial and error
    if N < 100:
        return 30
    patience = max(3, int(50 - math.log(N, 1.25)))
    return patience


def get_loss(model, device, data_loader, loss_function):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for data in data_loader:
            X, y = data[0].to(device), data[1].to(device)
            output = model(X)
            loss = loss_function(y, output)
            loss_total += loss.item()
    return loss_total / len(data_loader)


class CustomDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class Forecaster:
    """NBEATS Timeseries Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    MODEL_NAME = "NBEATS_Timeseries_Forecaster"

    def __init__(
        self,
        backcast_length: int,
        forecast_length: int,
        num_exog: int = 0,
        num_generic_stacks: int = 2,
        nb_blocks_per_stack: int = 2,
        thetas_dim_per_stack: int = 16,
        hidden_layer_units: int = 32,
        share_weights_in_stack: bool = False,
        batch_size: int = 32,
        **kwargs,
    ):
        """Construct a new NBEATS Forecaster.
        Args:
           backcast_length (int): Encoding (history) length.
           forecast_length (int): Decoding (forecast) length.
           num_exog (int, optional): Number of exogenous variables.
                                           Defaults to 0.
           num_generic_stacks (int, optional): Number of generic stacks.
                                           Defaults to 2.
           nb_blocks_per_stack (int, optional): Number of blocks per stack.
                                           Defaults to 2.
           thetas_dim_per_stack (int, optional): Number of expansion coefficients.
                                           Defaults to 16.
           hidden_layer_units (int, optional): Hidden layer units.
                                           Defaults to 32.
           share_weights_in_stack (boolean, optional): Whether to share weights within stacks.
                                           Defaults to False.
        """
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.num_exog = num_exog
        self.num_generic_stacks = num_generic_stacks
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim_per_stack = thetas_dim_per_stack
        self.hidden_layer_units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.batch_size = batch_size
        self.model = self.build_model()
        self.model.compile(optimizer="adam", loss="mse")
        self.loss = "mse"
        self.learning_rate = 1e-4

        self.criterion = MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.print_period = 1

    def build_model(self):
        """Build the NBEATS model."""
        return NBeatsNet(
            device=device,
            backcast_length=self.backcast_length,
            forecast_length=self.forecast_length,
            nb_blocks_per_stack=self.nb_blocks_per_stack,
            thetas_dim=[self.thetas_dim_per_stack] * self.num_generic_stacks,
            share_weights_in_stack=self.share_weights_in_stack,
            hidden_layer_units=self.hidden_layer_units,
        )

    def _get_X_y_and_E(self, data: np.ndarray, is_train: bool = True) -> np.ndarray:
        """Extract X (historical target series), y (forecast window target) and
        E (exogenous series) from given array of shape [N, T, D]

        When is_train is True, data contains both history and forecast windows.
        When False, only history is contained.
        """
        N, T, D = data.shape
        if D != 1 + self.num_exog:
            raise ValueError(
                f"Training data expected to have {self.num_exog} exogenous variables. "
                f"Found {D-1}"
            )
        if is_train:
            if T != self.backcast_length + self.forecast_length:
                raise ValueError(
                    f"Training data expected to have {self.backcast_length + self.forecast_length}"
                    f" length on axis 1. Found length {T}"
                )
            X = data[:, : self.backcast_length, :1]
            y = data[:, self.backcast_length :, :1]
            if D > 1:
                E = data[:, : self.backcast_length, 1:]
            else:
                E = None
        else:
            # for inference
            if T < self.backcast_length:
                raise ValueError(
                    f"Inference data length expected to be >= {self.backcast_length}"
                    f" on axis 1. Found length {T}"
                )
            X = data[:, -self.backcast_length :, :1]
            y = None
            if D > 1:
                E = data[:, -self.backcast_length :, 1:]
            else:
                E = None
        return X, y, E

    def _train_on_data(self, data, validation_split=0.1, max_epochs=500):
        """Train the model on the given data.

        Args:
            data (pandas.DataFrame): The training data.
        """
        print(data)
        X, y, E = self._get_X_y_and_E(data, is_train=True)
        print(X.shape, "shape of x")
        print(y.shape, "shape of y")
        loss_to_monitor = "loss" if validation_split is None else "val_loss"
        patience = get_patience_factor(X.shape[0])
        X_val, y_val = None, None
        if validation_split > 0:
            val_size = int(validation_split * X.shape[1])
            X_val = X[-val_size:, :, :]
            y_val = y[-val_size:, :, :]
            X = X[:-val_size, :, :]
            y = y[:-val_size, :, :]

        history = self.model.fit(
            x_train=[X, E] if E is not None else X,
            y_train=y,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=self.batch_size,
            patience=patience,
        )
        # recompile the model to reset the optimizer; otherwise re-training slows down
        return history

    # def _get_X_and_y(self, data: np.ndarray, is_train: bool = True) -> np.ndarray:
    #     """Extract X (historical target series), y (forecast window target)
    #     When is_train is True, data contains both history and forecast windows.
    #     When False, only history is contained.
    #     """
    #     N, T, D = data.shape
    #     if D != self.feat_dim:
    #         raise ValueError(
    #             f"Training data expected to have {self.feat_dim} feature dim. "
    #             f"Found {D}"
    #         )
    #     if is_train:
    #         if T != self.encode_len + self.decode_len:
    #             raise ValueError(
    #                 f"Training data expected to have {self.encode_len + self.decode_len}"
    #                 f" length on axis 1. Found length {T}"
    #             )
    #         X = data[:, : self.encode_len, :]
    #         y = data[:, self.encode_len :, 0]
    #     else:
    #         # for inference
    #         if T < self.encode_len:
    #             raise ValueError(
    #                 f"Inference data length expected to be >= {self.encode_len}"
    #                 f" on axis 1. Found length {T}"
    #             )
    #         X = data[:, -self.encode_len :, :]
    #         y = None
    #     return X, y

    def fit(
        self,
        train_data: np.ndarray,
        pre_training_data: Union[np.ndarray, None] = None,
        validation_split: Union[float, None] = 0.15,
        max_epochs: int = 2000,
    ):
        """Fit the Forecaster to the training data.
        A separate Prophet model is fit to each series that is contained
        in the data.

        Args:
            data (pandas.DataFrame): The features of the training data.
        """
        if pre_training_data is not None:
            print("Conducting pretraining...")
            pretraining_history = self._train_on_data(
                data=pre_training_data,
                validation_split=validation_split,
                max_epochs=max_epochs,
            )

        print("Training on main data...")
        history = self._train_on_data(
            data=train_data,
            validation_split=validation_split,
            max_epochs=max_epochs,
        )
        self._is_trained = True
        return history

    # def fit(self, train_data, valid_data, max_epochs=250, verbose=1):
    #     train_X, train_y = self._get_X_and_y(train_data, is_train=True)
    #     if valid_data is not None:
    #         valid_X, valid_y = self._get_X_and_y(valid_data, is_train=True)
    #     else:
    #         valid_X, valid_y = None, None

    #     self.batch_size = max(1, min(train_X.shape[0] // 8, 256))
    #     print(f"batch_size = {self.batch_size}")

    #     patience = get_patience_factor(train_X.shape[0])
    #     print(f"{patience=}")

    #     train_X, train_y = torch.FloatTensor(train_X), torch.FloatTensor(train_y)
    #     train_dataset = CustomDataset(train_X, train_y)
    #     train_loader = DataLoader(
    #         dataset=train_dataset, batch_size=int(self.batch_size), shuffle=True
    #     )

    #     if valid_X is not None and valid_y is not None:
    #         valid_X, valid_y = torch.FloatTensor(valid_X), torch.FloatTensor(valid_y)
    #         valid_dataset = CustomDataset(valid_X, valid_y)
    #         valid_loader = DataLoader(
    #             dataset=valid_dataset, batch_size=int(self.batch_size), shuffle=True
    #         )
    #     else:
    #         valid_loader = None

    #     losses = self._run_training(
    #         train_loader,
    #         valid_loader,
    #         max_epochs,
    #         use_early_stopping=True,
    #         patience=patience,
    #         verbose=verbose,
    #     )
    #     return losses

    # def _run_training(
    #     self,
    #     train_loader,
    #     valid_loader,
    #     max_epochs,
    #     use_early_stopping=True,
    #     patience=10,
    #     verbose=1,
    # ):

    #     best_loss = 1e7
    #     losses = []
    #     min_epochs = 10
    #     for epoch in range(max_epochs):
    #         self.net.train()
    #         for data in train_loader:
    #             X, y = data[0].to(device), data[1].to(device)
    #             # Feed Forward
    #             preds = self.net(X)
    #             # Loss Calculation
    #             loss = self.criterion(y, preds)
    #             # Clear the gradient buffer (we don't want to accumulate gradients)
    #             self.optimizer.zero_grad()
    #             # Backpropagation
    #             loss.backward()
    #             # Weight Update: w <-- w - lr * gradient
    #             self.optimizer.step()

    #         current_loss = loss.item()

    #         if use_early_stopping:
    #             # Early stopping
    #             if valid_loader is not None:
    #                 current_loss = get_loss(
    #                     self.net, device, valid_loader, self.criterion
    #                 )
    #             losses.append({"epoch": epoch, "loss": current_loss})
    #             if current_loss < best_loss:
    #                 trigger_times = 0
    #                 best_loss = current_loss
    #             else:
    #                 trigger_times += 1
    #                 if trigger_times >= patience and epoch >= min_epochs:
    #                     if verbose == 1:
    #                         print(f"Early stopping after {epoch=}!")
    #                     return losses
    #         else:
    #             losses.append({"epoch": epoch, "loss": current_loss})
    #         # Show progress
    #         if verbose == 1:
    #             if epoch % self.print_period == 0 or epoch == max_epochs - 1:
    #                 print(
    #                     f"Epoch: {epoch+1}/{max_epochs}, loss: {np.round(current_loss, 5)}"
    #                 )

    #     return losses

    def predict(self, data):
        X = self._get_X_y_and_E(data, is_train=False)[0]
        pred_X = torch.FloatTensor(X)
        # Initialize dataset and dataloader with only X
        pred_dataset = CustomDataset(pred_X)
        pred_loader = DataLoader(
            dataset=pred_dataset, batch_size=self.batch_size, shuffle=False
        )
        with torch.no_grad():
            all_preds = []
            for data in pred_loader:
                X = data.to(device)
                preds = self.model.predict(X).squeeze()
                all_preds.append(preds)

        preds = np.concatenate(all_preds, axis=0)
        preds = np.expand_dims(preds, axis=-1)
        if len(preds.shape) == 2:
            preds = np.expand_dims(preds, axis=0)
        return preds

    def evaluate(self, test_data):
        """Evaluate the model and return the loss and metrics"""
        x_test, y_test = self._get_X_y_and_E(test_data, is_train=True)
        if self.net is not None:
            x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test)
            dataset = CustomDataset(x_test, y_test)
            data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
            current_loss = get_loss(self.net, device, data_loader, self.criterion)
            return current_loss

    def save(self, model_dir_path: str) -> None:
        """Save the forecaster to disk.

        Args:
            model_dir_path (str): The dir path to which to save the model.
        """
        if self.model is None:
            raise NotFittedError("Model is not fitted yet.")
        model_params = {
            "backcast_length": self.backcast_length,
            "forecast_length": self.forecast_length,
            "num_exog": self.num_exog,
            "num_generic_stacks": self.num_generic_stacks,
            "nb_blocks_per_stack": self.nb_blocks_per_stack,
            "thetas_dim_per_stack": self.thetas_dim_per_stack,
            "hidden_layer_units": self.hidden_layer_units,
            "share_weights_in_stack": self.share_weights_in_stack,
        }
        joblib.dump(model_params, os.path.join(model_dir_path, MODEL_PARAMS_FNAME))
        torch.save(
            self.model.state_dict(), os.path.join(model_dir_path, MODEL_WTS_FNAME)
        )

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded forecaster.
        """
        if not os.path.exists(model_dir_path):
            raise FileNotFoundError(f"Model dir {model_dir_path} does not exist.")
        model_params = joblib.load(os.path.join(model_dir_path, MODEL_PARAMS_FNAME))
        forecaster_model = cls(**model_params)
        forecaster_model.model.load_state_dict(
            torch.load(os.path.join(model_dir_path, MODEL_WTS_FNAME))
        )
        return forecaster_model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.MODEL_NAME}"


def train_predictor_model(
    train_data: np.ndarray,
    valid_data: np.ndarray,
    forecast_length: int,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the forecaster model.

    Args:
        train_data (np.ndarray): The train split from training data.
        valid_data (np.ndarray): The valid split of training data.
        forecast_length (int): Length of forecast window.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """
    model = Forecaster(
        backcast_length=train_data.shape[1] - forecast_length,
        forecast_length=forecast_length,
        num_exog=train_data.shape[2] - 1,
        **hyperparameters,
    )
    model.fit(
        train_data=train_data,
    )
    return model


def predict_with_model(model: Forecaster, test_data: np.ndarray) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (np.ndarray): The test input data for forecasting.

    Returns:
        np.ndarray: The forecast.
    """
    return model.predict(test_data)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
