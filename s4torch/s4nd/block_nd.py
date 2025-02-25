from typing import Type, Optional

import torch
from torch import nn

from s4torch.aux.adapters import TemporalAdapter
from s4torch.aux.residual import Residual, SequentialWithResidual
from s4torch.s41d.block import _make_norm
from s4torch.s4nd.layer_nd import S4NDLayer


class S4NDBlock(nn.Module):
    """S4Nd Block for multidimensional inputs (e.g., images, videos).

    Args:
        d_model (int): number of internal features
        n (int): dimensionality of the state representation
        shape (tuple[int]): shape of the input signal
        Other args are the same as the regular S4Block.
    """
    def __init__(
        self,
        d_model: int,
        n: int,
        shape: tuple[int, ...],
        p_dropout: float = 0.0,
        activation: Type[nn.Module] = nn.GELU,
        norm_type: Optional[str] = "layer",
        norm_strategy: str = "post",
        pooling: Optional[nn.AvgPool1d | nn.MaxPool1d] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.shape = shape
        self.p_dropout = p_dropout
        self.activation = activation
        self.norm_type = norm_type
        self.norm_strategy = norm_strategy
        self.pooling = pooling

        self.pipeline = SequentialWithResidual(
            (
                _make_norm(d_model, norm_type=norm_type)
                if norm_strategy in ("pre", "both")
                else nn.Identity()
            ),
            S4NDLayer(d_model, n=n, seq_shape=shape),
            activation(),
            nn.Dropout(p_dropout),
            nn.Linear(d_model, d_model, bias=True),
            Residual(),
            (
                _make_norm(d_model, norm_type=norm_type)
                if norm_strategy in ("post", "both")
                else nn.Identity()
            ),
            TemporalAdapter(pooling) if pooling else nn.Identity(),
            nn.Dropout(p_dropout)
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass for N-dimensional input."""
        return self.pipeline(u)


if __name__ == "__main__":
    # Test S4NdBlock
    N = 32  # dimensionality of states
    d_model = 128  # number of internal features
    seq_shape = (16, 16, 3)  # Shape of the sequence (e.g., temporal, spatial, channels)

    # Create a 28x28 pixel image with d_model channels
    u = torch.randn(1, *seq_shape, d_model)  # Shape: [BATCH, HEIGHT, WIDTH, D_INPUT]

    # Initialize the S4ND layer
    s4nd_block = S4NDBlock(d_model, n=N, shape=seq_shape).to(device=u.device)

    # Forward pass through the S4ND layer
    output = s4nd_block(u)

    # Ensure the output shape matches the input shape
    assert output.shape == u.shape  # Shape: [1, 28, 28, 128] -> [1, 28, 28, 128]
    print("Output shape:", output.shape)

