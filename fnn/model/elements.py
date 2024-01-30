import math
import torch
from torch import nn
from itertools import chain
from functools import reduce
from collections import deque
from .parameters import Parameter, ParameterList
from .modules import Module, ModuleList
from .utils import add, cat_groups


def nonlinearity(nonlinear=None):
    """Adapted from:
    https://github.com/deepmind/deepmind-research/blob/cb555c241b20c661a3e46e5d1eb722a0a8b0e8f4/nfnets/base.py#L101

    Parameters
    ----------
    nonlinear : str | None
        "gelu" | "silu" | "elu" | "tanh" | None

    Returns
    -------
    Callable[[Tensor], Tensor]
        nonlinear transform
    float
        scaling factor to preserve variance
    """
    if nonlinear == "gelu":
        return nn.GELU(approximate="none"), 1.7015043497085571

    elif nonlinear == "silu":
        return nn.SiLU(inplace=False), 1.7881293296813965

    elif nonlinear == "elu":
        return nn.ELU(alpha=1.0, inplace=False), 1.2716004848480225

    elif nonlinear == "tanh":
        return nn.Tanh(), 1.5939117670059204

    elif nonlinear is None:
        return nn.Identity(), 1.0

    else:
        raise NotImplementedError(f'"{nonlinear}" not implemented')


class Dropout(Module):
    def __init__(self, p=0):
        """
        Parameters
        ----------
        p : float
            dropout probability between 0 and 1
        """
        super().__init__()
        self.p = p

    def _reset(self):
        self.mask = None

    @property
    def p(self):
        if self.training:
            return self._p
        else:
            return 0

    @p.setter
    def p(self, p):
        assert 0 <= p < 1
        self._p = float(p)
        self.scale = 1 / (1 - self._p)
        self.mask = None

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            [N, C, H, W]

        Returns
        -------
        Tensor
            [N, C, H, W]
        """
        if not self.p:
            return x

        if self.mask is None:
            N, C, _, _ = x.shape
            rand = torch.rand([N, C], device=x.device)
            self.mask = (rand > self.p) * self.scale

        return torch.einsum("N C H W , N C -> N C H W", x, self.mask)

    def extra_repr(self):
        return f"p={self.p:.3g}"


class FlatDropout(Dropout):
    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            [N, C]

        Returns
        -------
        Tensor
            [N, C]
        """
        if not self.p:
            return x

        if self.mask is None:
            N, C = x.shape
            rand = torch.rand([N, C], device=x.device)
            self.mask = (rand > self.p) * self.scale

        return x * self.mask


class Conv(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        in_groups=1,
        out_groups=1,
        streams=1,
        temporal=1,
        spatial=1,
        stride=1,
        pad="zeros",
        gain=1,
        bias=0,
        eps=1e-5,
        wnorm=True,
    ):
        """
        Parameters
        ----------
        in_channels : int
            input channels per stream, must be divisible by in_groups
        out_channels : int
            output channels per stream, must be divisible by in_groups and out_groups
        in_groups : int
            input groups per stream
        out_groups : int
            output groups per stream
        streams : int
            number of streams
        temporal : int
            temporal kernel size
        spatial : int
            spatial kernel size
        stride : int
            spatial stride
        pad : str | None
            spatial padding mode -- 'zeros' | 'replicate' | None
        gain : float | None
            initial gain value
        bias : float | None
            initial bias value
        eps : float
            small value for numerical stability
        wnorm : bool
            enable weight norm
        """
        if in_channels % in_groups != 0:
            raise ValueError("Input channels must be divisible by input groups")

        if out_channels % in_groups != 0:
            raise ValueError("Output channels must be divisible by input groups")

        if out_channels % out_groups != 0:
            raise ValueError("Output channels must be divisible by output groups")

        if (spatial - stride) % 2 != 0:
            raise ValueError("Incompatible spatial_size and stride")

        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.in_groups = int(in_groups)
        self.out_groups = int(out_groups)
        self.streams = int(streams)
        self.temporal = int(temporal)
        self.spatial = int(spatial)
        self.stride = int(stride)

        self.pad = None if pad is None else str(pad)
        self.padding = 0 if self.pad is None else (self.spatial - self.stride) // 2

        self.gain = gain is not None
        self.bias = bias is not None

        self.init_gain = float(gain) if self.gain else None
        self.init_bias = float(bias) if self.bias else None

        self.eps = float(eps)
        self.wnorm = bool(wnorm)

        if self.pad is None:
            self.pad_fn = lambda x: x
        elif self.pad == "zeros":
            self.pad_fn = lambda x: nn.functional.pad(x, pad=[self.padding] * 4)
        elif self.pad == "replicate":
            self.pad_fn = lambda x: nn.functional.pad(x, pad=[self.padding] * 4, mode="replicate")
        else:
            raise ValueError("Invalid pad mode")

        shape = [
            self.out_channels,
            self.in_channels // self.in_groups,
            self.temporal,
            self.spatial,
            self.spatial,
        ]
        self.fan_in = math.prod(shape[1:])
        bound = math.sqrt(1 / self.fan_in) if self.wnorm else math.sqrt(3 / self.fan_in)

        def param():
            weight = torch.zeros(shape)
            nn.init.uniform_(weight, -bound, bound)
            return Parameter(weight)

        self.weights = ParameterList([param() for _ in range(self.streams)])
        self.weights.norm_dim = [1, 2, 3, 4]

        self.fan_out = self.out_channels // self.out_groups

        if self.wnorm and self.gain:
            gain = lambda: Parameter(torch.full([self.out_groups, self.fan_out], self.init_gain))
            self.gains = ParameterList([gain() for _ in range(streams)])
            self.gains.decay = False
            self.gains.norm_dim = 1

        if self.bias:
            bias = lambda: Parameter(torch.full([self.out_groups, self.fan_out], self.init_bias))
            self.biases = ParameterList([bias() for _ in range(streams)])
            self.biases.decay = False
            self.biases.norm_dim = 1

        self.past = dict()

    def _reset(self):
        self.past.clear()

    def weight(self, stream=None):
        """
        Parameters
        ----------
        stream : int | None
            specific stream (int) or all streams (None)
        """
        if stream is None:
            weights = [self.weight(stream) for stream in range(self.streams)]
            return torch.cat(weights, dim=0)

        elif self.wnorm:
            weight = self.weights[stream]
            var, mean = torch.var_mean(weight, dim=[1, 2, 3, 4], keepdim=True, unbiased=False)

            scale = (var * self.fan_in + self.eps).pow(-0.5)
            if self.gain:
                scale = scale * self.gains[stream].view_as(scale)

            return (weight - mean) * scale

        else:
            return self.weights[stream]

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : 4D Tensor
            [N, C, H, W] -- stream is int
                or
            [N, S*C, H, W] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, C', H, W] -- stream is int
                or
            [N, S*C', H, W] -- stream is None
        """
        x = self.pad_fn(x)

        if self.past:
            assert self.past["stream"] == stream
            weight = self.past["weight"]
            history = self.past["history"]

        else:
            self.past["stream"] = stream
            self.past["weight"] = weight = self.weight(stream)

            if self.temporal > 1:
                start = x if self.pad == "replicate" else torch.zeros_like(x)
                self.past["history"] = history = deque([start] * self.temporal, maxlen=self.temporal)
            else:
                self.past["history"] = history = None

        if history is None:
            x = x.unsqueeze(dim=2)
        else:
            history.append(x)
            x = torch.stack(list(history), dim=2)

        if stream is None:
            groups = self.in_groups * self.streams
            bias = torch.cat([_.flatten() for _ in self.biases]) if self.bias else None
        else:
            groups = self.in_groups
            bias = self.biases[stream].flatten() if self.bias else None

        y = nn.functional.conv3d(input=x, weight=weight, bias=bias, groups=groups, stride=self.stride)
        return y.squeeze(dim=2)

    def extra_repr(self):
        s = "{streams} x {inp}->{out}, gain={gain}, bias={bias}"
        s = s.format(
            streams=self.streams,
            inp=self.in_channels,
            out=self.out_channels,
            stride=self.stride,
            gain=self.gain,
            bias=self.bias,
        )
        if self.padding:
            s += f", pad={self.pad}"
        if self.stride > 1:
            s += f", stride={self.stride}"
        return s


class Linear(Conv):
    def __init__(
        self,
        in_features,
        out_features,
        in_groups=1,
        out_groups=1,
        streams=1,
        gain=1,
        bias=0,
        eps=1e-5,
        wnorm=True,
    ):
        """
        Parameters
        ----------
        in_features : int
            input features per stream, must be divisible by in_groups
        out_features : int
            output features per stream, must be divisible by in_groups and out_groups
        in_groups : int
            input groups per stream
        out_groups : int
            output groups per stream
        streams : int
            number of streams
        gain : float | None
            initial gain value
        bias : float | None
            initial bias value
        eps : float
            small value for numerical stability
        wnorm : bool
            enable weight norm
        """
        super().__init__(
            in_channels=in_features,
            out_channels=out_features,
            in_groups=in_groups,
            out_groups=out_groups,
            streams=streams,
            gain=gain,
            bias=bias,
            eps=eps,
            wnorm=wnorm,
        )

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : 2D Tensor
            [N, F] -- stream is int
                or
            [N, S*F] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, F'] -- stream is int
                or
            [N, S*F'] -- stream is None
        """
        return super().forward(x[:, :, None, None], stream=stream)[:, :, 0, 0]


class InterGroup(Module):
    def __init__(self, in_channels, out_channels, groups, streams=1, gain=1, eps=1e-5):
        """
        Parameters
        ----------
        in_channels : int
            in channels per stream, must be divisible by groups
        out_channels : int
            out channels per stream, must be divisible by groups
        groups : int
            groups per stream
        streams : int
            number of streams
        gain : float | None
            output gain
        eps : float
            small value for numerical stability
        """
        if in_channels % groups != 0:
            raise ValueError("Input channels must be divisible by groups")

        if out_channels % groups != 0:
            raise ValueError("Output channels must be divisible by groups")

        if not groups > 1:
            raise ValueError("Groups must be greater than 1")

        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.streams = int(streams)
        self.gain = gain is not None
        self.init_gain = float(gain) if self.gain else None
        self.eps = float(eps)

        self.group_in = self.in_channels // self.groups
        self.group_out = self.out_channels // self.groups

        self.fan_in = self.group_in * (self.groups - 1)

        def weight(bound):
            weight = torch.zeros([self.groups, self.groups - 1, self.group_out, self.group_in])
            nn.init.uniform_(weight, -bound, bound)
            return Parameter(weight)

        bound = self.fan_in**-0.5
        self.weights = ParameterList([weight(bound) for _ in range(self.streams)])
        self.weights.norm_dim = (1, 3)

        mask = ~torch.eye(self.groups, dtype=torch.bool)[:, :, None, None]
        self.register_buffer("mask", mask)

        zero = torch.zeros([self.groups, self.groups, self.group_out, self.group_in])
        self.register_buffer("zero", zero)

        if self.gain:
            gain = lambda: Parameter(torch.full([self.groups, self.group_out], self.init_gain))
            self.gains = ParameterList([gain() for _ in range(streams)])
            self.gains.decay = False
            self.gains.norm_dim = 1

        self.past = dict()

    def _reset(self):
        self.past.clear()

    def weight(self, stream=None):
        """
        Parameters
        ----------
        stream : int | None
            specific stream (int) or all streams (None)
        """
        if stream is None:
            weights = [self.weight(stream) for stream in range(self.streams)]
            return torch.cat(weights, dim=0)

        else:
            weight = self.weights[stream]
            var, mean = torch.var_mean(weight, dim=(1, 3), keepdim=True, unbiased=False)

            scale = (var * self.fan_in + self.eps).pow(-0.5)
            if self.gain:
                scale = scale * self.gains[stream].view_as(scale)

            weight = (weight - mean) * scale
            weight = torch.masked_scatter(self.zero, self.mask, weight)
            weight = torch.einsum("O I A B -> O A I B", weight)

            return weight.reshape(self.out_channels, self.in_channels, 1, 1)

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : 4D Tensor
            [N, C, H, W] -- stream is int
                or
            [N, S*C, H, W] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, C', H, W] -- stream is int
                or
            [N, S*C', H, W] -- stream is None
        """
        if self.past:
            assert self.past["stream"] == stream
            weight = self.past["weight"]
        else:
            self.past["stream"] = stream
            self.past["weight"] = weight = self.weight(stream)

        if stream is None:
            groups = self.streams
        else:
            groups = 1

        return nn.functional.conv2d(input=x, weight=weight, groups=groups)

    def extra_repr(self):
        s = "{streams} x {inp}->{out}, groups={groups}, gain={gain}"
        return s.format(
            streams=self.streams,
            inp=self.in_channels,
            out=self.out_channels,
            groups=self.groups,
            gain=self.gain,
        )
        return s


class Accumulate(ModuleList):
    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : Sequence[Tensor]
            tensors to project onto a common space
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            tensor in projection space
        """
        assert len(x) == len(self)
        return add([module(_, stream=stream) for module, _ in zip(self, x)])


class Lstm(Module):
    def __init__(
        self,
        in_features,
        out_features,
        streams,
        wnorm=True,
        dropout=0,
        init_input=-1,
        init_forget=1,
    ):
        """
        Parameters
        ----------
        in_features : int
            input features per stream (I)
        out_channels : int
            output channels per stream (O)
        streams : int
            number of streams
        wnorm : bool
            enable weight norm
        dropout : float
            dropout probability -- [0, 1)
        init_input : float
            initial input gate bias
        init_forget : float
            initial forget gate bias
        """
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.streams = int(streams)
        self.wnorm = bool(wnorm)

        self._dropout = float(dropout)
        self.drop_x = FlatDropout(p=self._dropout)
        self.drop_h = FlatDropout(p=self._dropout)

        self.proj_i = Linear(
            in_features=self.in_features + self.out_features,
            out_features=self.out_features,
            streams=self.streams,
            wnorm=self.wnorm,
            bias=float(init_input),
        )
        self.proj_f = Linear(
            in_features=self.in_features + self.out_features,
            out_features=self.out_features,
            streams=self.streams,
            wnorm=self.wnorm,
            bias=float(init_forget),
        )
        self.proj_g = Linear(
            in_features=self.in_features + self.out_features,
            out_features=self.out_features,
            streams=self.streams,
            wnorm=self.wnorm,
        )
        self.proj_o = Linear(
            in_features=self.in_features + self.out_features,
            out_features=self.out_features,
            streams=self.streams,
            wnorm=self.wnorm,
        )
        self.past = dict()

    def _restart(self):
        self.dropout(p=self._dropout)

    def _reset(self):
        self.past.clear()

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : 4D Tensor
            [N, I] -- stream is int
                or
            [N, S*I] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, O] -- stream is int
                or
            [N, S*O] -- stream is None
        """
        if stream is None:
            S = self.streams
        else:
            S = 1

        if self.past:
            h = self.past["h"]
            c = self.past["c"]
        else:
            h = c = torch.zeros([1, S * self.out_features], device=self.device)

        x = self.drop_x(x)
        xh = cat_groups([x, h], groups=S, expand=True)

        i = torch.sigmoid(self.proj_i(xh, stream=stream))
        f = torch.sigmoid(self.proj_f(xh, stream=stream))
        g = torch.tanh(self.proj_g(xh, stream=stream))
        o = torch.sigmoid(self.proj_o(xh, stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)
        h = self.drop_h(h)

        self.past["c"] = c
        self.past["h"] = h

        return h


class Mlp(Module):
    def __init__(self, in_features, out_features, out_wnorms, out_nonlinears, streams=1):
        """
        Parameters
        ----------
        in_features : int
            input features
        out_features : Sequence[int]
            output features
        out_wnorms : Sequence[bool]
            output weight norms
        out_nonlinears : Sequence[int]
            output nonlinearities
        streams : int
            number of streams
        """
        assert len(out_features) == len(out_wnorms) == len(out_nonlinears)
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = list(map(int, out_features))
        self.out_wnorms = list(map(bool, out_wnorms))
        self.out_nonlinears = list(out_nonlinears)
        self.streams = int(streams)

        self.linears = ModuleList()
        self.nonlinears = ModuleList()
        self.gammas = []

        in_features = self.in_features

        for features, wnorm, nonlinear in zip(
            self.out_features,
            self.out_wnorms,
            self.out_nonlinears,
        ):
            linear = Linear(in_features=in_features, out_features=features, streams=streams, wnorm=wnorm)
            in_features = features
            self.linears.append(linear)

            nonlinear, gamma = nonlinearity(nonlinear=nonlinear)
            self.nonlinears.append(nonlinear)
            self.gammas.append(gamma)

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : 2D Tensor
            [N, F] -- stream is int
                or
            [N, S*F] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, F'] -- stream is int
                or
            [N, S*F'] -- stream is None
        """
        for linear, nonlinear, gamma in zip(
            self.linears,
            self.nonlinears,
            self.gammas,
        ):
            x = linear(x, stream=stream)
            x = nonlinear(x)
            x = x * gamma

        return x
