"""

    S4 Model

"""

from __future__ import annotations

from typing import Optional, Type

import torch
from torch import nn

from s4torch.aux.encoders import StandardEncoder
from s4torch.dsp.utils import next_pow2
from s4torch.s43d.block3d import S43DBlock


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
    l_max: int,
    pool_kernel: Optional[int | tuple[int]],
) -> list[tuple[int, int]]:
    ppk = _parse_pool_kernel(pool_kernel)

    schedule = list()
    for depth in range(n_blocks + 1):
        l_max_next = max(2, l_max // ppk)
        pool_ok = l_max_next > ppk
        schedule.append((l_max, pool_ok))
        l_max = l_max_next
    return schedule


class S43DModel(nn.Module):
    """S4 Model.

    High-level implementation of the S4 model which:

        1. Encodes the input using a linear layer
        2. Applies ``1..n_blocks`` S4 blocks
        3. Decodes the output of step 2 using another linear layer

    Args:
        d_input (int): number of input features
        d_model (int): number of internal features
        d_output (int): number of features to return
        n_blocks (int): number of S4 blocks to construct
        n (int): dimensionality of the state representation
        l_max (int): length of input signal
        wavelet_tform (bool): if ``True`` encode signal using a
            continuous wavelet transform (CWT).
        collapse (bool): if ``True`` average over time prior to
            decoding the result of the S4 block(s). (Useful for
            classification tasks.)
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
        d_input: int,
        d_model: int,
        d_output: int,
        n_blocks: int,
        n: int,
        seq_shape: tuple,
        wavelet_tform: bool = False,
        collapse: bool = False,
        p_dropout: float = 0.0,
        activation: Type[nn.Module] = nn.GELU,
        norm_type: Optional[str] = "layer",
        norm_strategy: str = "post",
        pooling: Optional[nn.AvgPool1d | nn.MaxPool1d] = None,
    ) -> None:
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output
        self.n_blocks = n_blocks
        self.n = n
        self.seq_shape = seq_shape
        self.wavelet_tform = wavelet_tform
        self.collapse = collapse
        self.p_dropout = p_dropout
        self.norm_type = norm_type
        self.norm_strategy = norm_strategy
        self.pooling = pooling

        if wavelet_tform:
            from s4torch.dsp.cwt import Cwt, CwtWithAdapter

            self.encoder = CwtWithAdapter(
                Cwt(next_pow2(self.l_max)),
                d_model=self.d_model,
            )
        else:
            self.encoder = StandardEncoder(self.d_input, d_model=self.d_model)

        self.decoder = nn.Linear(self.d_model, self.d_output)
        self.blocks = nn.ModuleList(
            [
                S43DBlock(
                    d_model=d_model,
                    n=n,
                    seq_shape=seq_shape,
                    p_dropout=p_dropout,
                    activation=activation,
                    norm_type=norm_type,
                    norm_strategy=norm_strategy,
                    pooling=pooling,
                ) for _ in range(n_blocks)
            ]
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            u (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_INPUT]``

        Returns:
            y (torch.Tensor): a tensor of the form ``[BATCH, D_OUTPUT]`` if ``collapse``
                is ``True`` and ``[BATCH, SEQ_LEN // (POOL_KERNEL ** n_block), D_INPUT]``
                otherwise, where ``POOL_KERNEL`` is the kernel size of the ``pooling``
                layer. (Note that ``POOL_KERNEL=1`` if ``pooling`` is ``None``.)

        """
        y = self.encoder(u)
        for block in self.blocks:
            y = block(y)    # [B, S, D] -> [B, S, D]
        return self.decoder(y.mean(dim=tuple(range(1, u.dim() - 1))) if self.collapse else y)  # [B, S = 784, D = 10] -> [B, D_out = 10]


if __name__ == "__main__":
    from experiments.utils import count_parameters

    N = 64
    d_input = 1
    d_model = 32
    d_output = 4
    n_blocks = 6
    seq_shape = (8, 8, 16)

    u = torch.randn(1, *seq_shape, d_input)

    s43dmodel = S43DModel(
        d_input=d_input,
        d_model=d_model,
        d_output=d_output,
        n_blocks=n_blocks,
        n=N,
        seq_shape=seq_shape,
        collapse=False,
        wavelet_tform=False,
    )
    print(f"S4Model Params: {count_parameters(s43dmodel):,}")

    loss = nn.MSELoss()
    target = torch.randn(1, *seq_shape, d_output)
    optimizer = torch.optim.Adam(s43dmodel.parameters(), lr=0.02)

    for e in range(500):
        optimizer.zero_grad()
        output = s43dmodel(u)
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()
        print(f"Epoch {e + 1}: Loss={loss_value.item()}")

    assert s43dmodel(u).shape == (u.shape[0], *u.shape[1:-1], s43dmodel.d_output)
