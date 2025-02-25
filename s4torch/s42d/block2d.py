"""

    S4 Block

"""

from __future__ import annotations

from typing import Optional, Type

import torch
from torch import nn
from torch.nn import LayerNorm

from s4torch.aux.adapters import TemporalAdapter
from s4torch.aux.residual import Residual, SequentialWithResidual
from s4torch.s42d.layer2d import S42DLayer


def _make_norm(d_model: int, norm_type: Optional[str]) -> nn.Module:
    if norm_type is None:
        return nn.Identity()
    elif norm_type == "layer":
        return nn.LayerNorm(d_model)
    elif norm_type == "batch":
        return TemporalAdapter(nn.BatchNorm1d(d_model))
    else:
        raise ValueError(f"Unsupported norm type '{norm_type}'")


class S42DBlock(nn.Module):
    """S4 Block.

    Applies ``S4Layer()``, followed by an activation
    function, dropout, linear layer, skip connection and
    layer normalization.

    Args:
        d_model (int): number of internal features
        n (int): dimensionality of the state representation
        l_max (int): length of input signal
        p_dropout (float): probability of elements being set to zero
        activation (Type[nn.Module]): activation function to use after
            ``S4Layer()``.
        norm_type (str, optional): type of normalization to use.
            Options: ``batch``, ``layer``, ``None``.
        norm_strategy (str): position of normalization relative to ``S4Layer()``.
            Must be "pre" (before ``S4Layer()``), "post" (after ``S4Layer()``)
            or "both" (before and after ``S4Layer()``).
        pooling (nn.AvgPool1d, nn.MaxPool1d, optional): pooling method to use
            following each ``S4Block()``.

    """

    def __init__(
        self,
        d_model: int,
        n: int,
        seq_shape: tuple,
        p_dropout: float = 0.0,
        activation: Type[nn.Module] = nn.GELU,
        norm_type: Optional[str] = "layer",
        norm_strategy: str = "post",
        pooling: Optional[nn.AvgPool1d | nn.MaxPool1d] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.seq_shape = seq_shape
        self.p_dropout = p_dropout
        self.activation = activation
        self.norm_type = norm_type
        self.norm_strategy = norm_strategy
        self.pooling = pooling

        if norm_strategy not in ("pre", "post", "both"):
            raise ValueError(f"Unexpected norm_strategy, got '{norm_strategy}'")

        self.pipeline = SequentialWithResidual(
            (
                _make_norm(d_model, norm_type=norm_type)
                if norm_strategy in ("pre", "both")
                else nn.Identity()
            ),
            S42DLayer(d_model, n=n, seq_shape=seq_shape),
            activation(),
            nn.Dropout(p_dropout),
            nn.Linear(d_model, d_model, bias=True),
            Residual(),
            LayerNorm(d_model),
            TemporalAdapter(pooling) if pooling else nn.Identity(),
            nn.Dropout(p_dropout),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            u (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_INPUT]``

        Returns:
            y (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_OUTPUT]``

        """
        return self.pipeline(u)


if __name__ == "__main__":
    from experiments.utils import count_parameters

    N = 64
    d_input = 1
    d_model = 128  # Number of internal features (add encoder and decoder layers to get to input and output number of channels)
    seq_shape = (28, 28)

    u = torch.randn(1, *seq_shape, d_input)

    s42dblock = S42DBlock(d_model, n=N, seq_shape=seq_shape, norm_type="batch").to(device=u.device)
    print(f"S4Block Params: {count_parameters(s42dblock):,}")

    loss = nn.MSELoss()
    target = torch.randn_like(u)
    optimizer = torch.optim.Adam(s42dblock.parameters(), lr=0.05)

    for e in range(100):
        optimizer.zero_grad()
        output = s42dblock(u)
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()
        print(f"Epoch {e + 1}: Loss={loss_value.item()}")

    assert s42dblock(u).shape == u.shape
