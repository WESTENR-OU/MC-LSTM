import numpy as np
import torch
from torch import nn
from typing import Tuple, List
import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # This line checks if GPU is available


class MassConservingLSTM(nn.Module):
    """ Pytorch implementation of Mass-Conserving LSTMs. """

    def __init__(self, in_dim: int, aux_dim: int, out_dim: int,
                 in_gate: nn.Module = None, out_gate: nn.Module = None,
                 redistribution: nn.Module = None, time_dependent: bool = True,
                 batch_first: bool = False):
        """
        Parameters
        ----------
        in_dim : int
            The number of mass inputs.
        aux_dim : int
            The number of auxiliary inputs.
        out_dim : int
            The number of cells or, equivalently, outputs.
        in_gate : nn.Module, optional
            A module computing the (normalised!) input gate.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `in_dim` x `out_dim` matrix for every sample.
            Defaults to a time-dependent softmax input gate.
        out_gate : nn.Module, optional
            A module computing the output gate.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `out_dim` vector for every sample.
        redistribution : nn.Module, optional
            A module computing the redistribution matrix.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `out_dim` x `out_dim` matrix for every sample.
        time_dependent : bool, optional
            Use time-dependent gates if `True` (default).
            Otherwise, use only auxiliary inputs for gates.
        batch_first : bool, optional
            Expects first dimension to represent samples if `True`,
            Otherwise, first dimension is expected to represent timesteps (default).
        """
        super().__init__()
        self.in_dim = in_dim
        self.aux_dim = aux_dim
        self.out_dim = out_dim
        self._seq_dim = 1 if batch_first else 0

        gate_inputs = aux_dim + out_dim + in_dim

        # initialize gates
        if out_gate is None:
            self.out_gate = _Gate(in_features=gate_inputs, out_features=out_dim)
        if in_gate is None:
            self.in_gate = _NormalizedGate(in_features=gate_inputs,
                                           out_shape=(in_dim, out_dim),
                                           normalizer="normalized_sigmoid")
        if redistribution is None:
            self.redistribution = _NormalizedGate(in_features=gate_inputs,
                                                  out_shape=(out_dim, out_dim),
                                                  normalizer="normalized_relu")
        self._reset_parameters()

    @property
    def batch_first(self) -> bool:
        return self._seq_dim != 0

    def reset_parameters(self, out_bias: float = -3.):
        """
        Parameters
        ----------
        out_bias : float, optional
            The initial bias value for the output gate (default to -3).
        """
        self.redistribution.reset_parameters(bias_init=nn.init.eye_)
        self.in_gate.reset_parameters(bias_init=nn.init.zeros_)
        self.out_gate.reset_parameters(
            bias_init=lambda b: nn.init.constant_(b, val=out_bias)
        )

    def _reset_parameters(self, out_bias: float = -3.):
        nn.init.constant_(self.out_gate.fc.bias, val=out_bias)

    def forward(self, xm, xa, state=None):
        xm = xm.unbind(dim=self._seq_dim)
        xa = xa.unbind(dim=self._seq_dim)

        if state is None:
            state = self.init_state(len(xa[0]))

        hs, cs = [], []
        for xm_t, xa_t in zip(xm, xa):
            # xm xa shape: [batchsize, 1] (i.e., [256,1])
            h, state = self._step(xm_t, xa_t, state)  # h.shape=[256,16], state.shape=[256,16]
            hs.append(h)
            cs.append(state)

        hs = torch.stack(hs, dim=self._seq_dim)  # [256, 365, 16]
        cs = torch.stack(cs, dim=self._seq_dim)  # [256, 365, 16]
        return hs, cs

    @torch.no_grad()
    def init_state(self, batch_size: int):
        """ Create the default initial state. """
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.out_dim, device=device)

    def _step(self, xt_m, xt_a, c):
        """ Make a single time step in the MCLSTM. """
        # in this version of the MC-LSTM all available data is used to derive the gate activations. Cell states
        # are L1-normalized so that growing cell states over the sequence don't cause problems in the gates.
        features = torch.cat([xt_m, xt_a, c / (c.norm(1) + 1e-5)], dim=-1)

        # compute gate activations
        i = self.in_gate(features)
        r = self.redistribution(features)
        o = self.out_gate(features)

        # distribute incoming mass over the cell states
        m_in = torch.matmul(xt_m.unsqueeze(-2), i).squeeze(-2)

        # reshuffle the mass in the cell states using the redistribution matrix
        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)

        # compute the new mass states
        m_new = m_in + m_sys

        # return the outgoing mass and subtract this value from the cell states.
        return o * m_new, (1 - o) * m_new

class _Gate(nn.Module):
    """Utility class to implement a standard sigmoid gate"""

    def __init__(self, in_features: int, out_features: int):
        super(_Gate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the normalised gate"""
        return torch.sigmoid(self.fc(x))


class _NormalizedGate(nn.Module):
    """Utility class to implement a gate with normalised activation function"""

    def __init__(self, in_features: int, out_shape: Tuple[int, int], normalizer: str):
        super(_NormalizedGate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_shape[0] * out_shape[1])
        self.out_shape = out_shape

        if normalizer == "normalized_sigmoid":
            self.activation = nn.Sigmoid()
        elif normalizer == "normalized_relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                f"Unknown normalizer {normalizer}. Must be one of {'normalized_sigmoid', 'normalized_relu'}")
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the normalized gate"""
        h = self.fc(x).view(-1, *self.out_shape)
        return torch.nn.functional.normalize(self.activation(h), p=1, dim=-1)


def train_epoch(model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch.
    :param model: A torch.nn.Module implementing the MC-LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param loader: A PyTorch DataLoader, providing the trainings
        data in mini batches.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    # set model to train mode (important for dropout)
    loss_list = []
    model.train()
    pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        xm = xs[..., 0:1]
        xa = xs[..., 1:2]
        # get model predictions
        m_out, c = model(xm, xa)  # [batch size, seq length, hidden size]
        # y_hat = m_out[:, -1:].sum(dim=-1)
        output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)  # trash cell excluded [batch size, seq length, 1]
        # y_hat = output.transpose(0, 1)
        y_hat = output[:, -1, :]
        # print('**** output shape: ', output.shape, "y_hat.shape: ", y_hat.shape)
        # print('***** ys shape: ', ys.shape)
        # calculate loss
        loss = loss_func(y_hat, ys)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        # calculate gradients
        loss.backward()
        loss_list.append(loss)
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
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
            xm = xs[..., 0:1]
            xa = xs[..., 1:2]
            # get model predictions
            m_out, c = model(xm, xa)
            # y_hat = m_out[:, -1:].sum(dim=-1)
            output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)  # trash cell excluded [batch size, seq length, 1]
            # y_hat = output.transpose(0, 1)
            y_hat = output[:, -1, :]
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
            xm = xs[..., 0:1]
            xa = xs[..., 1:2]
            # get model predictions
            m_out, c = model(xm, xa)
            output = m_out[:, -1, :]  # [batch size, 1, hidden sizes]
            obs.append(ys)
            hidden.append(output)

    return torch.cat(obs), torch.cat(hidden)


def get_cell_states(model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.

    :return: Two torch Tensors, containing the observations and
        model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    cell = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # push data to GPU (if available)
            xs = xs.to(DEVICE)
            # get model predictions
            xm = xs[..., 0:1]
            xa = xs[..., 1:2]
            # get model predictions
            m_out, c = model(xm, xa)
            cell_state = c[:, -1, :].sum(dim=-1, keepdim=True)
            obs.append(ys)
            cell.append(cell_state)

    return torch.cat(obs), torch.cat(cell)


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
