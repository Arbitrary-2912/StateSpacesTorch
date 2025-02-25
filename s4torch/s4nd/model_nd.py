from __future__ import annotations

from typing import Optional, Type

import torch
from torch import nn

from s4torch.aux.encoders import StandardEncoder
from s4torch.s4nd.block_nd import S4NDBlock
from s4torch.dsp.utils import next_pow2


def _parse_pool_kernel(pool_kernel: Optional[int | tuple[int]]) -> int:
    if pool_kernel is None:
        return 1
    elif isinstance(pool_kernel, tuple):
        return pool_kernel[0]
    elif isinstance(pool_kernel, int):
        return pool_kernel
    else:
        raise TypeError(f"Unable to parse `pool_kernel`, got {pool_kernel}")


def _seq_length_schedule(
    n_blocks: int,
    shape_max: tuple[int, ...],
    pool_kernel: Optional[int | tuple[int]],
) -> list[tuple[tuple[int, ...], bool]]:
    ppk = _parse_pool_kernel(pool_kernel)

    schedule = list()
    for depth in range(n_blocks + 1):
        shape_next = tuple(max(2, s // ppk) for s in shape_max)
        pool_ok = all(s_next > ppk for s_next in shape_next)
        schedule.append((shape_max, pool_ok))
        shape_max = shape_next
    return schedule


class S4NDModel(nn.Module):
    """S4 Model for multidimensional inputs (e.g., images, videos).

    High-level implementation of the S4 model which:
        1. Encodes the input using a linear layer
        2. Applies ``1..n_blocks`` S4 blocks (S4NdBlock)
        3. Decodes the output of step 2 using another linear layer

    Args:
        d_input (int): number of input features (channels)
        d_model (int): number of internal features
        d_output (int): number of features to return
        n_blocks (int): number of S4 blocks to construct
        n (int): dimensionality of the state representation
        seq_shape (tuple[int, ...]): shape of the input signal (e.g., height, width, channels)
        wavelet_tform (bool): if ``True`` encode signal using a continuous wavelet transform (CWT).
        collapse (bool): if ``True`` average over time prior to decoding the result of the S4 block(s).
        p_dropout (float): probability of elements being set to zero
        activation (Type[nn.Module]): activation function to use after ``S4NdBlock()``.
        norm_type (str, optional): type of normalization to use. Options: ``batch``, ``layer``, ``None``.
        norm_strategy (str): position of normalization relative to ``S4NdBlock()``. Must be "pre", "post", or "both".
        pooling (nn.Module, optional): pooling method to use following each ``S4NdBlock()``.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        n_blocks: int,
        n: int,
        seq_shape: tuple[int, ...],
        wavelet_tform: bool = False,
        collapse: bool = False,
        p_dropout: float = 0.0,
        activation: Type[nn.Module] = nn.GELU,
        norm_type: Optional[str] = "layer",
        norm_strategy: str = "post",
        pooling: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output
        self.n_blocks = n_blocks
        self.n = n
        self.shape_max = seq_shape
        self.wavelet_tform = wavelet_tform
        self.collapse = collapse
        self.p_dropout = p_dropout
        self.norm_type = norm_type
        self.norm_strategy = norm_strategy
        self.pooling = pooling

        # Determine the sequence length schedule based on shape_max
        *self.seq_len_schedule, (self.seq_len_out, _) = _seq_length_schedule(
            n_blocks=n_blocks,
            shape_max=seq_shape,
            pool_kernel=None if self.pooling is None else self.pooling.kernel_size,
        )

        if wavelet_tform:
            from s4torch.dsp.cwt import Cwt, CwtWithAdapter

            self.encoder = CwtWithAdapter(
                Cwt(next_pow2(self.shape_max[-2])),  # Handle height/width dimension for wavelet transform
                d_model=self.d_model,
            )
        else:
            self.encoder = StandardEncoder(self.d_input, d_model=self.d_model)

        self.decoder = nn.Linear(self.d_model, self.d_output)

        # Initialize S4NdBlocks for each stage
        self.blocks = nn.ModuleList(
            [
                S4NDBlock(
                    d_model=d_model,
                    n=n,
                    shape=shape,
                    p_dropout=p_dropout,
                    activation=activation,
                    norm_type=norm_type,
                    norm_strategy=norm_strategy,
                    pooling=pooling if pooling and pool_ok else None,
                )
                for (shape, pool_ok) in self.seq_len_schedule
            ]
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            u (torch.Tensor): a tensor of the form ``[BATCH, *SHAPE, D_INPUT]`` where ``*SHAPE`` is a multidimensional shape.

        Returns:
            y (torch.Tensor): a tensor of the form ``[BATCH, D_OUTPUT]`` if ``collapse`` is ``True``
            and ``[BATCH, *SEQUENCE_SHAPE, D_INPUT]`` otherwise.
        """
        # Encoding the input signal
        y = self.encoder(u)

        # Pass through each S4NdBlock
        for block in self.blocks:
            y = block(y)

        # Decoding the result (either collapsing or returning the full sequence)
        return self.decoder(y.mean(dim=tuple(range(1, u.dim() - 1))) if self.collapse else y)  # Collapse if needed


if __name__ == "__main__":
    N = 64
    d_input = 3  # Example: RGB channels
    d_model = 128
    d_output = 128
    n_blocks = 6
    seq_shape = (16, 16, 3)  # Example: (height, width, channels)

    u = torch.randn(1, *seq_shape, d_input)  # Shape: [1, 16, 16, 3, 128]

    s4model_nd = S4NDModel(
        d_input,
        d_model=d_model,
        d_output=d_output,
        n_blocks=n_blocks,
        n=N,
        seq_shape=seq_shape,
        collapse=False,
        wavelet_tform=False,
    )

    output = s4model_nd(u)
    print("Output shape:", output.shape)  # Shape after processing

