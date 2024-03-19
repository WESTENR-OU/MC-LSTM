import sys

sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
import tqdm
import time
from typing import Tuple

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # This line checks if GPU is available


class Model(nn.Module):
    """Implementation of a single layer LSTM network"""

    def __init__(self, hidden_size: int, dropout_rate: float = 0.0):
        """Initialize model

        :param hidden_size: Number of hidden units/LSTM cells
        :param dropout_rate: Dropout rate of the last fully connected
            layer. Default 0.0
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # create required layer
        self.lstm = nn.LSTM(input_size=2, hidden_size=self.hidden_size,
                            num_layers=1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Network.

        :param x: Tensor of shape [batch size, seq length, num features]
            containing the input data for the LSTM network.

        :return: Tensor containing the network predictions
        """
        output, (h_n, c_n) = self.lstm(x)
        # perform prediction only at the end of the input sequence
        pred = self.fc(self.dropout(h_n[-1, :, :]))
        pred = torch.relu(pred)
        return pred


def train_epoch(model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch.
    :param model: A torch.nn.Module implementing the LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param loader: A PyTorch DataLoader, providing the trainings
        data in mini batchSes.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    pbar.set_description(f"Epoch {epoch}")
    loss_list = []
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)

        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
        loss_list.append(loss)
    loss_ave = np.mean(torch.stack(loss_list).detach().numpy())
    return loss_ave


def eval_model(model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.

    :return: Two torch Tensors, containing the observations and
        model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # push data to GPU (if available)
            xs = xs.to(DEVICE)
            # get model predictions
            y_hat = model(xs)
            obs.append(ys)
            preds.append(y_hat)

    return torch.cat(obs), torch.cat(preds)

def get_hidden_nodes(model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.

    :return: Two torch Tensors, containing the observations and
        model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    hidden = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # push data to GPU (if available)
            xs = xs.to(DEVICE)
            # get model predictions
            y_hat = model(xs)
            obs.append(ys)
            hidden.append(y_hat)

        return torch.cat(obs), torch.cat(hidden)

def calc_nse(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator

    return nse_val


def calc_rmse(obs: np.array, sim: np.array) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: RMSE value.
    """
    # Only consider time steps where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # Check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    # Calculate RMSE
    rmse_val = np.sqrt(np.mean((sim - obs) ** 2))

    return rmse_val


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, sim, obs):
        return torch.sqrt(torch.mean((sim - obs) ** 2))


