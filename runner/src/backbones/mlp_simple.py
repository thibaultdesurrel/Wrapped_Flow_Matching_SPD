import torch
from torch import nn

from spd_learn.functional import sym_to_upper, vec_to_sym

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}


class MLP(nn.Module):
    """
    The MLP used in the paper of Collas and al.
    """

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        cond_dim: int,
        time_dim: int = 1,
        hidden_dim: int = 128,
        dtype=torch.float64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.time_emb_dim = time_dim
        if isinstance(dtype, str):
            self.dtype = DTYPE_MAP[dtype]
        else:
            self.dtype = dtype
        self.hidden_dim = hidden_dim

        layers = []
        layers.append(
            nn.Linear(
                self.input_dim + self.cond_dim + self.time_emb_dim,
                self.hidden_dim,
                dtype=self.dtype,
            )
        )
        layers.append(nn.SELU())
        for i in range(n_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, dtype=self.dtype))
            layers.append(nn.SELU())
        layers.append(nn.Linear(self.hidden_dim, self.input_dim, dtype=self.dtype))

        self.net = nn.Sequential(*layers)

    def forward(self, x, t, cond):
        x_vec = sym_to_upper(x).to(self.dtype)
        sz = x_vec.size()
        t = t.reshape(-1, 1).expand(x_vec.shape[0], 1)

        h = torch.cat([x_vec, cond, t], dim=-1).to(dtype=self.dtype)
        output_vec = self.net(h)
        output_vec = output_vec.reshape(*sz)
        return vec_to_sym(output_vec)
