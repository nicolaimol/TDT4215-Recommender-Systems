import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class PositionalEncoder(nn.Module):
    def __init__(self, dim_hid, num_pos=128):
        super(PositionalEncoder, self).__init__()

        self.register_buffer(
            'pos_table',
            self._get_sinusoid_encoding_table(num_pos, dim_hid)
        )

        self.pos_table = self._get_sinusoid_encoding_table(num_pos, dim_hid)

    def _get_sinusoid_encoding_table(self, n_position, dim_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / dim_hid) for hid_j in range(dim_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position + 1)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, x.size(1)].clone().detach()


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim_in, dim_hid, elu_func="gelu", dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()

        activation_dict = {
            # rectified linear unit
            'relu': torch.relu,
            # randomized rectified linear unit
            'rrelu': torch.rrelu,
            # relu 6 pad = 6
            'relu6': nn.ReLU6(),
            # Exponential linear unit
            'elu': nn.ELU(),
            # continuously differentiable exponential linear units
            'celu': nn.CELU(),
            # self-normalizing exponential linear units
            'selu': nn.SELU(),
            # gaussian error linear units
            'gelu': nn.GELU(),
            # parametric rectified linear units
            'prelu': nn.PReLU()
        }

        self.ff = nn.Sequential(
            nn.LayerNorm(dim_in, eps=1e-6),
            nn.Linear(dim_in, dim_hid),
            activation_dict[elu_func],
            nn.Linear(dim_hid, dim_in),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        residual = x
        output = self.ff(x)

        return residual + output
