import uuid

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.f2py.symbolic import as_real
from sympy import Line2D
from torch import nn
from torch.fft import rfft, rfft2, rfftn, irfft, irfft2, irfftn, ifft
from torch.nn import functional as F, init

from s4torch.s41d.layer import _log_step_initializer, _make_p_q_lambda, _make_omega_l, _cauchy_dot


def _non_circular_convolution_1d(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    ud = rfft(F.pad(u.float(), pad=(0, 0, 0, u.shape[1], 0, 0)), dim=1)
    Kd = rfft(F.pad(K.float(), pad=(0, u.shape[1])), dim=-1)
    return irfft(ud.transpose(-2, -1) * Kd)[..., :u.shape[1]].transpose(-2, -1).type_as(u)


def _non_circular_convolution_2d(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    ud = rfft2(F.pad(u.float(), pad=(0, 0, 0, u.shape[2], 0, u.shape[1])), dim=(1, 2))
    Kd = rfft2(F.pad(K.float(), pad=(0, u.shape[2], 0, u.shape[1], 0, 0)), dim=(-2, -1))
    return irfft2(ud.permute(0, 3, 1, 2) * Kd)[..., :u.shape[1], :u.shape[2]].permute(0, 2, 3, 1).type_as(u)


def _non_circular_convolution_nd(u: torch.Tensor, K: torch.Tensor, n: int) -> torch.Tensor:
    ud_pad_tuple = [0, 0]
    for i in range(n, 0, -1):
        ud_pad_tuple.extend([0, u.shape[i]])
    ud_pad_tuple = tuple(ud_pad_tuple)

    Kd_pad_tuple = []
    for i in range(n, 0, -1):
        Kd_pad_tuple.extend([0, u.shape[i]])
    Kd_pad_tuple.extend([0, 0])
    Kd_pad_tuple = tuple(Kd_pad_tuple)

    ud_dims = tuple(range(1, n + 1))
    Kd_dims = tuple(range(-n, 0))

    permutation_tuple = [0, n + 1]
    for i in range(1, n + 1):
        permutation_tuple.append(i)
    permutation_tuple = tuple(permutation_tuple)

    inverse_permutation_tuple = [0]
    for i in range(2, n + 2):
        inverse_permutation_tuple.append(i)
    inverse_permutation_tuple.append(1)
    inverse_permutation_tuple = tuple(inverse_permutation_tuple)

    slices = [slice(None)] * 2
    for i in range(n):
        slices.append(slice(0, u.shape[i + 1]))

    ud = rfftn(F.pad(u.float(), pad=ud_pad_tuple), dim=ud_dims)
    Kd = rfftn(F.pad(K.float(), pad=Kd_pad_tuple), dim=Kd_dims)

    return irfftn(ud.permute(permutation_tuple) * Kd)[tuple(slices)].permute(inverse_permutation_tuple).type_as(u)

class S4NDLayer(nn.Module):
    """S4ND Layer.

    Structured State Space for (Long) Sequences (S4ND) layer, generalizing to N-dimensional inputs.

    Args:
        d_model (int): number of internal features
        n (int): dimensionality of the state representation
        seq_shape (tuple): shape of the sequence (temporal, spatial, channels, etc.)
    """

    def __init__(self, d_model: int, n: int, seq_shape: tuple) -> None:
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.seq_shape = seq_shape  # sequence length (e.g., temporal or channel dimension)

        self._p = []
        self._q = []
        self._lambda_ = []
        self._omega_l = []
        self._B = []
        self._Ct = []
        self.log_step = []
        self.ifft_order = []

        for i in range(len(self.seq_shape)):
            p, q, lambda_ = map(lambda t: t.type(torch.complex64), _make_p_q_lambda(n))
            self._p.append(nn.Parameter(torch.view_as_real(p)))
            self._q.append(nn.Parameter(torch.view_as_real(q)))
            self._lambda_.append(nn.Parameter(torch.view_as_real(lambda_).unsqueeze(0).unsqueeze(1)))

            self.register_parameter(
                f"p_{i}",
                self._p[i]
            )
            self.register_parameter(
                f"q_{i}",
                self._q[i]
            )
            self.register_parameter(
                f"lambda_{i}",
                self._lambda_[i]
            )

            self.register_buffer(
                f"omega_l_{i}",
                tensor=_make_omega_l(self.seq_shape[i], dtype=torch.complex64),
            )
            self._omega_l.append(_make_omega_l(self.seq_shape[i], dtype=torch.complex64))

            self.register_buffer(
                f"ifft_order_{i}",
                tensor=torch.as_tensor(
                    [x if x == 0 else self.seq_shape[i] - x for x in range(self.seq_shape[i])],
                    dtype=torch.long,
                ),
            )
            self.ifft_order.append(torch.as_tensor(
                [x if x == 0 else self.seq_shape[i] - x for x in range(self.seq_shape[i])],
                dtype=torch.long,
            ))

            self._B.append(nn.Parameter(
                torch.view_as_real(init.xavier_normal_(torch.empty(d_model, n, dtype=torch.complex64)))
            ))
            self._Ct.append(nn.Parameter(
                torch.view_as_real(init.xavier_normal_(torch.empty(d_model, n, dtype=torch.complex64)))
            ))
            self.log_step.append(nn.Parameter(_log_step_initializer(torch.rand(d_model))))

            self.register_parameter(
                f"B_{i}",
                self._B[i]
            )
            self.register_parameter(
                f"Ct_{i}",
                self._Ct[i]
            )
            self.register_parameter(
                f"log_step_{i}",
                self.log_step[i]
            )

        self.D = nn.Parameter(torch.ones(*([1] * (len(seq_shape) + 1)), d_model))

    def _compute_roots(self, p: torch.Tensor, q: torch.Tensor, lambda_: torch.Tensor, omega_l: torch.Tensor, B: torch.Tensor, Ct: torch.Tensor, log_step: torch.Tensor) -> torch.Tensor:
        """Compute the roots of the polynomial defined by p and q."""
        a0, a1 = Ct.conj(), q.conj()
        b0, b1 = B, p
        step = log_step.exp()

        g = torch.outer(2.0 / step, (1.0 - omega_l) / (1.0 + omega_l))
        c = 2.0 / (1.0 + omega_l)
        cauchy_dot_denominator = g.unsqueeze(-1) - lambda_

        k00 = _cauchy_dot(a0 * b0, denominator=cauchy_dot_denominator)
        k01 = _cauchy_dot(a0 * b1, denominator=cauchy_dot_denominator)
        k10 = _cauchy_dot(a1 * b0, denominator=cauchy_dot_denominator)
        k11 = _cauchy_dot(a1 * b1, denominator=cauchy_dot_denominator)
        return c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

    def _compute_kernel_given_params(self, p: torch.Tensor, q: torch.Tensor, lambda_: torch.Tensor, omega_l: torch.Tensor, ifft_order: torch.Tensor, B: torch.Tensor, Ct: torch.Tensor, log_step: torch.Tensor, l: int) -> torch.Tensor:
        at_roots = self._compute_roots(p, q, lambda_, omega_l, B, Ct, log_step)
        out = ifft(at_roots, n=l, dim=-1)
        conv = torch.stack([i[ifft_order] for i in out]).real
        return conv.unsqueeze(0)

    def _compute_kernel(self) -> torch.Tensor:
        """Compute the kernel K used for convolution via outer product of 1D kernels."""
        kernel = self._compute_kernel_given_params(
            torch.view_as_complex(self._p[0]),
            torch.view_as_complex(self._q[0]),
            torch.view_as_complex(self._lambda_[0]),
            self._omega_l[0],
            self.ifft_order[0],
            torch.view_as_complex(self._B[0]),
            torch.view_as_complex(self._Ct[0]),
            self.log_step[0],
            self.seq_shape[0]
        )
        if len(self.seq_shape) == 1:
            return kernel
        running_contraction = "bhi"
        for i in range(1, len(self.seq_shape)):
            kernel_i = self._compute_kernel_given_params(
                torch.view_as_complex(self._p[i]),
                torch.view_as_complex(self._q[i]),
                torch.view_as_complex(self._lambda_[i]),
                self._omega_l[i],
                self.ifft_order[i],
                torch.view_as_complex(self._B[i]),
                torch.view_as_complex(self._Ct[i]),
                self.log_step[i],
                self.seq_shape[i]
            )
            kernel_contraction = f"bh{chr(ord(running_contraction[-1]) + 1)}"
            target_contraction = running_contraction + f"{chr(ord(running_contraction[-1]) + 1)}"
            kernel = torch.einsum(f"{running_contraction},{kernel_contraction}->{target_contraction}", kernel,
                                  kernel_i)
            running_contraction = target_contraction
        return kernel

    def convolve(self, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Non-circular N-D convolution."""
        n = u.ndim - 2
        if n == 1:
            return _non_circular_convolution_1d(u, K)
        elif n == 2:
            return _non_circular_convolution_2d(u, K)
        else:
            return _non_circular_convolution_nd(u, K, n)

    @property
    def K(self) -> torch.Tensor:  # noqa
        """K convolutional filter."""
        return self._compute_kernel()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass for N-dimensional inputs (e.g., 28x28 image).

        Args:
            u (torch.Tensor): a tensor of the form ``[BATCH, HEIGHT, WIDTH, D_INPUT]``

        Returns:
            y (torch.Tensor): a tensor of the form ``[BATCH, HEIGHT, WIDTH, D_OUTPUT]``

        """
        return self.convolve(u, K=self._compute_kernel()) + (self.D * u)


if __name__ == "__main__":
    N = 32  # dimensionality of states
    d_model = 128  # number of internal features
    seq_shape = tuple([768, 12])  # Shape of the sequence (e.g., temporal, spatial, channels)

    # Create a 28x28 pixel image with d_model channels
    u = torch.randn(1, *seq_shape, d_model)  # Shape: [BATCH, HEIGHT, WIDTH, D_INPUT]

    # Initialize the S4ND layer
    s4nd_layer = S4NDLayer(d_model, n=N, seq_shape=seq_shape).to(device=u.device)

    loss = nn.MSELoss()
    target = torch.randn_like(u)
    optimizer = torch.optim.Adam(s4nd_layer.parameters(), lr=0.01)

    for e in range(1000):
        optimizer.zero_grad()
        output = s4nd_layer(u)
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()

    # Forward pass through the S4ND layer
    output = s4nd_layer(u)

    # Ensure the output shape matches the input shape
    assert output.shape == u.shape  # Shape: [1, 28, 28, 128] -> [1, 28, 28, 128]
    print("Output shape:", output.shape)
