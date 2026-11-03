"""ConvLSTM forecaster: ConvLSTMCell -> stacked ConvLSTM encoder -> convolutional prediction head.

The cell follows Shi et al. (2015), "Convolutional LSTM Network", in the widely used
single-convolution form: input and hidden state are concatenated and one ``Conv2d`` produces all
four gates. Peephole connections are omitted.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell.

    Args:
        input_dim: Channels of the input frame.
        hidden_dim: Channels of the hidden and cell states.
        kernel_size: Odd ``(kh, kw)`` convolution kernel; "same" padding keeps H and W.
        bias: Whether the gate convolution has a bias.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: tuple[int, int] = (3, 3), bias: bool = True) -> None:
        super().__init__()
        kernel_size = tuple(kernel_size)
        if any(k % 2 == 0 for k in kernel_size):
            raise ValueError(f"kernel_size must be odd for same padding, got {kernel_size}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim, kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2), bias=bias)
        if bias:
            # A forget-gate bias of 1 is a standard trick that helps gradients flow early in training.
            with torch.no_grad():
                self.conv.bias.zero_()
                self.conv.bias[hidden_dim:2 * hidden_dim].fill_(1.0)

    def forward(self, input_tensor: torch.Tensor, cur_state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one time step.

        Args:
            input_tensor: ``(B, input_dim, H, W)``.
            cur_state: ``(h_cur, c_cur)``, each ``(B, hidden_dim, H, W)``.

        Returns:
            ``(h_next, c_next)``.
        """
        h_cur, c_cur = cur_state
        gates = self.conv(torch.cat([input_tensor, h_cur], dim=1))
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size: tuple[int, int], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-initialised ``(h, c)``."""
        shape = (batch_size, self.hidden_dim, *image_size)
        return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)


class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM over a ``(B, T, C, H, W)`` sequence.

    Args:
        input_dim: Input channels ``C``.
        hidden_dim: Hidden channels, either one int for every layer or a list with one per layer.
        kernel_size: Kernel for every layer, or a list with one per layer.
        num_layers: Number of stacked cells.
        bias: Gate-convolution bias.
        return_all_layers: If True, return outputs and states for every layer; otherwise only the last.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | Sequence[int],
        kernel_size: tuple[int, int] | Sequence[tuple[int, int]] = (3, 3),
        num_layers: int = 1,
        bias: bool = True,
        return_all_layers: bool = False,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dim) if isinstance(hidden_dim, (list, tuple)) else [hidden_dim] * num_layers
        if kernel_size and isinstance(kernel_size[0], (list, tuple)):
            kernel_sizes = [tuple(k) for k in kernel_size]
        else:
            kernel_sizes = [tuple(kernel_size)] * num_layers
        if not len(hidden_dims) == len(kernel_sizes) == num_layers:
            raise ValueError("hidden_dim and kernel_size lists must have num_layers entries")

        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.return_all_layers = return_all_layers
        self.cells = nn.ModuleList(
            ConvLSTMCell(input_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i], kernel_sizes[i], bias)
            for i in range(num_layers))

    def forward(
        self, x: torch.Tensor, hidden_state: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    ) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run the sequence through every layer.

        Args:
            x: ``(B, T, C, H, W)``.
            hidden_state: Optional initial ``(h, c)`` per layer; zeros by default.

        Returns:
            ``(layer_outputs, last_states)``. ``layer_outputs[l]`` has shape ``(B, T, hidden_l, H, W)``
            and ``last_states[l] = (h_T, c_T)``. Both contain only the last layer unless
            ``return_all_layers`` is set.
        """
        B, T, _, H, W = x.shape
        if hidden_state is None:
            hidden_state = [cell.init_hidden(B, (H, W), x.device, x.dtype) for cell in self.cells]

        layer_outputs: list[torch.Tensor] = []
        last_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        cur_input = x
        for layer, cell in enumerate(self.cells):
            h, c = hidden_state[layer]
            outputs = []
            for t in range(T):
                h, c = cell(cur_input[:, t], (h, c))
                outputs.append(h)
            cur_input = torch.stack(outputs, dim=1)
            layer_outputs.append(cur_input)
            last_states.append((h, c))

        if not self.return_all_layers:
            return layer_outputs[-1:], last_states[-1:]
        return layer_outputs, last_states


class PredictionHead(nn.Module):
    """Map the final hidden state to ``t_out`` future frames.

    A 1x1 convolution projects ``hidden_dim -> t_out * out_channels`` (direct multi-horizon
    forecasting). An optional stack of 3x3 convolutions then adds a residual correction that
    sharpens the spatial structure of the output.
    """

    def __init__(self, hidden_dim: int, out_channels: int, t_out: int = 1, refine_layers: int = 0, refine_channels: int = 32) -> None:
        super().__init__()
        self.t_out, self.out_channels = t_out, out_channels
        n_out = t_out * out_channels
        self.proj = nn.Conv2d(hidden_dim, n_out, kernel_size=1)
        self.refine: nn.Module | None = None
        if refine_layers > 0:
            layers: list[nn.Module] = []
            ch = n_out + hidden_dim
            for _ in range(refine_layers - 1):
                layers += [nn.Conv2d(ch, refine_channels, 3, padding=1), nn.ReLU(inplace=True)]
                ch = refine_channels
            layers.append(nn.Conv2d(ch, n_out, 3, padding=1))
            self.refine = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``(B, hidden_dim, H, W) -> (B, t_out, out_channels, H, W)``."""
        out = self.proj(h)
        if self.refine is not None:
            out = out + self.refine(torch.cat([out, h], dim=1))
        B, _, H, W = out.shape
        return out.view(B, self.t_out, self.out_channels, H, W)


class ConvLSTMForecaster(nn.Module):
    """ConvLSTM encoder plus prediction head.

    Args:
        input_dim: Input channels.
        output_dim: Target channels.
        t_out: Number of future frames to predict.
        hidden_dim, kernel_size, num_layers, bias: Passed to :class:`ConvLSTM`.
        refine_layers, refine_channels: Passed to :class:`PredictionHead`.
        residual_channels: Optional positions, within the *input* channels, of each target channel.
            When given, the network predicts the change from the last observed frame, i.e.
            ``y_hat = x[:, -1, residual_channels] + head(h_T)``. This is an optional experiment: in
            early tests with plain min-max scaling it tended to stall at the persistence solution,
            so it is off by default.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        t_out: int = 1,
        hidden_dim: int | Sequence[int] = 64,
        kernel_size: tuple[int, int] = (3, 3),
        num_layers: int = 2,
        bias: bool = True,
        refine_layers: int = 0,
        refine_channels: int = 32,
        residual_channels: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, bias)
        self.head = PredictionHead(self.encoder.hidden_dims[-1], output_dim, t_out, refine_layers, refine_channels)
        if residual_channels is not None and len(residual_channels) != output_dim:
            raise ValueError("residual_channels needs one entry per output channel")
        self.register_buffer(
            "residual_index",
            torch.tensor(list(residual_channels), dtype=torch.long) if residual_channels is not None else None,
            persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T_in, C_in, H, W) -> (B, T_out, C_out, H, W)``."""
        _, last_states = self.encoder(x)
        out = self.head(last_states[-1][0])
        if self.residual_index is not None:
            out = out + x[:, -1:, self.residual_index]
        return out


def build_model(cfg: dict[str, Any], input_dim: int, output_dim: int, residual_channels: Sequence[int] | None = None) -> ConvLSTMForecaster:
    """Instantiate :class:`ConvLSTMForecaster` from the ``model`` and ``data`` sections of the config."""
    m = cfg["model"]
    return ConvLSTMForecaster(
        input_dim=input_dim,
        output_dim=output_dim,
        t_out=cfg["data"]["t_out"],
        hidden_dim=m["hidden_dim"],
        kernel_size=tuple(m["kernel_size"]),
        num_layers=m["num_layers"],
        bias=m.get("bias", True),
        refine_layers=m.get("refine_layers", 0),
        refine_channels=m.get("refine_channels", 32),
        residual_channels=residual_channels if m.get("residual", False) else None,
    )


def residual_positions(input_channels: Sequence[int], target_channels: Sequence[int]) -> list[int] | None:
    """Position of each target channel within the input channels, or None if any target is not an input."""
    if not all(t in input_channels for t in target_channels):
        return None
    return [list(input_channels).index(t) for t in target_channels]
