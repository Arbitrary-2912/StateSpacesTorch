import torch
from torch import nn
from torch.fft import rfft2, irfft2, ifft
from torch.nn import functional as F, init

from s4torch.s41d.layer import _log_step_initializer, _make_p_q_lambda, _make_omega_l, _cauchy_dot


def _non_circular_convolution_2d(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    ud = rfft2(F.pad(u.float(), pad=(0, 0, 0, u.shape[2], 0, u.shape[1])), dim=(1, 2))
    Kd = rfft2(F.pad(K.float(), pad=(0, u.shape[2], 0, u.shape[1], 0, 0)), dim=(-2, -1))
    return irfft2(ud.permute(0, 3, 1, 2) * Kd)[..., :u.shape[1], :u.shape[2]].permute(0, 2, 3, 1).type_as(u)


class S42DLayer(nn.Module):
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

        # Kernel parameters for 1st dimension
        p, q, lambda_ = map(lambda t: t.type(torch.complex64), _make_p_q_lambda(n))
        self._p0 = nn.Parameter(torch.view_as_real(p))
        self._q0 = nn.Parameter(torch.view_as_real(q))
        self._lambda_0 = (nn.Parameter(torch.view_as_real(lambda_).unsqueeze(0).unsqueeze(1)))

        self.register_parameter(
            f"p_{0}",
            self._p0
        )
        self.register_parameter(
            f"q_{0}",
            self._q0
        )
        self.register_parameter(
            f"lambda_{0}",
            self._lambda_0
        )

        self.register_buffer(
            f"omega_l_{0}",
            tensor=_make_omega_l(self.seq_shape[0], dtype=torch.complex64),
        )
        self._omega_l_0 = _make_omega_l(self.seq_shape[0], dtype=torch.complex64)

        self.register_buffer(
            f"ifft_order_{0}",
            tensor=torch.as_tensor(
                [x if x == 0 else self.seq_shape[0] - x for x in range(self.seq_shape[0])],
                dtype=torch.long,
            ),
        )
        self.ifft_order_0 = torch.as_tensor(
            [x if x == 0 else self.seq_shape[0] - x for x in range(self.seq_shape[0])],
            dtype=torch.long,
        )

        self._B0 = nn.Parameter(
            torch.view_as_real(init.xavier_normal_(torch.empty(d_model, n, dtype=torch.complex64)))
        )
        self._Ct0 = nn.Parameter(
            torch.view_as_real(init.xavier_normal_(torch.empty(d_model, n, dtype=torch.complex64)))
        )
        self.log_step0 = nn.Parameter(_log_step_initializer(torch.rand(d_model)))

        self.register_parameter(
            f"B_{0}",
            self._B0
        )
        self.register_parameter(
            f"Ct_{0}",
            self._Ct0
        )
        self.register_parameter(
            f"log_step_{0}",
            self.log_step0
        )

        # Kernel parameters for 2nd dimension
        p, q, lambda_ = map(lambda t: t.type(torch.complex64), _make_p_q_lambda(n))
        self._p1 = nn.Parameter(torch.view_as_real(p))
        self._q1 = nn.Parameter(torch.view_as_real(q))
        self._lambda_1 = (nn.Parameter(torch.view_as_real(lambda_).unsqueeze(0).unsqueeze(1)))

        self.register_parameter(
            f"p_{1}",
            self._p1
        )
        self.register_parameter(
            f"q_{1}",
            self._q1
        )
        self.register_parameter(
            f"lambda_{1}",
            self._lambda_1
        )

        self.register_buffer(
            f"omega_l_{1}",
            tensor=_make_omega_l(self.seq_shape[1], dtype=torch.complex64),
        )
        self._omega_l_1 = _make_omega_l(self.seq_shape[1], dtype=torch.complex64)

        self.register_buffer(
            f"ifft_order_{1}",
            tensor=torch.as_tensor(
                [x if x == 0 else self.seq_shape[1] - x for x in range(self.seq_shape[1])],
                dtype=torch.long,
            ),
        )
        self.ifft_order_1 = torch.as_tensor(
            [x if x == 0 else self.seq_shape[1] - x for x in range(self.seq_shape[1])],
            dtype=torch.long,
        )

        self._B1 = nn.Parameter(
            torch.view_as_real(init.xavier_normal_(torch.empty(d_model, n, dtype=torch.complex64)))
        )
        self._Ct1 = nn.Parameter(
            torch.view_as_real(init.xavier_normal_(torch.empty(d_model, n, dtype=torch.complex64)))
        )
        self.log_step1 = nn.Parameter(_log_step_initializer(torch.rand(d_model)))

        self.register_parameter(
            f"B_{1}",
            self._B1
        )
        self.register_parameter(
            f"Ct_{1}",
            self._Ct1
        )
        self.register_parameter(
            f"log_step_{1}",
            self.log_step1
        )

        # LTI Skip Connection
        self.D = nn.Parameter(torch.ones(*([1] * (len(seq_shape) + 1)), d_model))

    def _compute_roots(self, p: torch.Tensor, q: torch.Tensor, lambda_: torch.Tensor, omega_l: torch.Tensor,
                       B: torch.Tensor, Ct: torch.Tensor, log_step: torch.Tensor) -> torch.Tensor:
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

    def _compute_kernel_given_params(self, p: torch.Tensor, q: torch.Tensor, lambda_: torch.Tensor,
                                     omega_l: torch.Tensor, ifft_order: torch.Tensor, B: torch.Tensor, Ct: torch.Tensor,
                                     log_step: torch.Tensor, l: int) -> torch.Tensor:
        at_roots = self._compute_roots(p, q, lambda_, omega_l, B, Ct, log_step)
        out = ifft(at_roots, n=l, dim=-1)
        conv = torch.stack([i[ifft_order] for i in out]).real
        return conv.unsqueeze(0)

    def _compute_kernel(self) -> torch.Tensor:
        """Compute the kernel K used for convolution via outer product of 1D kernels."""
        kernel_d0 = self._compute_kernel_given_params(
            torch.view_as_complex(self._p0),
            torch.view_as_complex(self._q0),
            torch.view_as_complex(self._lambda_0),
            self._omega_l_0,
            self.ifft_order_0,
            torch.view_as_complex(self._B0),
            torch.view_as_complex(self._Ct0),
            self.log_step0,
            self.seq_shape[0]
        )

        kernel_d1 = self._compute_kernel_given_params(
            torch.view_as_complex(self._p1),
            torch.view_as_complex(self._q1),
            torch.view_as_complex(self._lambda_1),
            self._omega_l_1,
            self.ifft_order_1,
            torch.view_as_complex(self._B1),
            torch.view_as_complex(self._Ct1),
            self.log_step1,
            self.seq_shape[1]
        )

        return kernel_d0.unsqueeze(-1) * kernel_d1.unsqueeze(-2)

    def convolve(self, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Non-circular N-D convolution."""
        return _non_circular_convolution_2d(u, K)

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
    seq_shape = tuple([28, 28])  # Shape of the sequence (e.g., temporal, spatial) Note channels is handled in d_input

    # Create a 28x28 pixel image with d_model channels
    u = torch.randn(1, *seq_shape, d_model)  # Shape: [BATCH, HEIGHT, WIDTH, D_INPUT]

    # Initialize the S4ND layer
    s4nd_layer = S42DLayer(d_model, n=N, seq_shape=seq_shape).to(device=u.device)

    loss = nn.MSELoss()
    target = torch.randn_like(u)
    optimizer = torch.optim.Adam(s4nd_layer.parameters(), lr=0.05)

    for e in range(100):
        optimizer.zero_grad()
        output = s4nd_layer(u)
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()
        print(f"Epoch {e + 1}: Loss={loss_value.item()}")

    # Forward pass through the S4ND layer
    output = s4nd_layer(u)

    # Ensure the output shape matches the input shape
    assert output.shape == u.shape  # Shape: [1, 28, 28, 128] -> [1, 28, 28, 128]
    print("Output shape:", output.shape)
