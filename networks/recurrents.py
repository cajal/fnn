import torch
from torch import nn

from .containers import Module
from .elements import Dropout, Conv
from .utils import to_groups_2d


class Recurrent(Module):
    @property
    def channels(self):
        raise NotImplementedError()

    @property
    def streams(self):
        raise NotImplementedError()

    @property
    def scale(self):
        raise NotImplementedError()

    def add_input(self, channels):
        """
        Parameters
        ----------
        channels : int
            input channels
        """
        raise NotImplementedError()

    def forward(self, inputs, dropout=0):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            shapes = [n, c, h, w]
        dropout : float
            dropout probability

        Returns
        -------
        Tensor
            shape = [n, c', h // scale, w // scale]
        """
        raise NotImplementedError()


class RvT(Recurrent):
    def __init__(self, channels, kernel_size, groups=1, streams=1):
        """
        Parameters
        ----------
        channels : int
            recurrent channels
        kernel_size : int
            spatial kernel size
        groups : int
            recurrent channel groups
        streams : int
            number of streams
        """
        super().__init__()

        self._channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.groups = int(groups)
        self._streams = int(streams)

        init = (channels / groups / streams) ** -0.5
        param = lambda: nn.Parameter(torch.full([channels // streams], init))
        self.tau = nn.ParameterList([param() for _ in range(streams)])

        self.drop = Dropout(
            drop_dim=[4, 5],
            reduce_dim=[3],
        )

        self.proj_x = Conv(channels=channels, groups=groups, streams=streams, gain=False, bias=False)
        if self.groups > 1:
            self.proj_x.add_intergroup()

        self.proj_q = Conv(channels=channels, groups=groups, streams=streams, gain=False, bias=False).add_input(
            channels=channels * 2, groups=groups, kernel_size=kernel_size
        )
        self.proj_k = Conv(channels=channels, groups=groups, streams=streams, gain=False, bias=False).add_input(
            channels=channels * 2, groups=groups, kernel_size=kernel_size, pad=False
        )
        self.proj_v = Conv(channels=channels, groups=groups, streams=streams, gain=False, bias=False).add_input(
            channels=channels * 2, groups=groups, kernel_size=kernel_size, pad=False
        )

        self.proj_i = (
            Conv(channels=channels, groups=groups, streams=streams)
            .add_input(channels=channels, groups=groups)
            .add_input(channels=channels * 2, groups=groups, kernel_size=kernel_size)
        )
        self.proj_f = (
            Conv(channels=channels, groups=groups, streams=streams)
            .add_input(channels=channels, groups=groups)
            .add_input(channels=channels * 2, groups=groups, kernel_size=kernel_size)
        )
        self.proj_g = (
            Conv(channels=channels, groups=groups, streams=streams)
            .add_input(channels=channels, groups=groups)
            .add_input(channels=channels * 2, groups=groups, kernel_size=kernel_size)
        )
        self.proj_o = (
            Conv(channels=channels, groups=groups, streams=streams)
            .add_input(channels=channels, groups=groups)
            .add_input(channels=channels * 2, groups=groups, kernel_size=kernel_size)
        )

        self._past = dict()

    def _reset(self):
        self._past.clear()

    @property
    def channels(self):
        return self._channels

    @property
    def streams(self):
        return self._streams

    @property
    def scale(self):
        return 1

    def add_input(self, channels):
        """
        Parameters
        ----------
        channels : int
            input channels
        """
        self.proj_x.add_input(channels=channels)

    def forward(self, inputs, stream=None, dropout=0):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            shapes = [n, c, h, w] -- stream is None
                or
            shapes = [n, c // s, h, w] -- stream is not None
        stream : int | None
            specific stream index (int) or all streams (None)
        dropout : float
            dropout probability

        Returns
        -------
        Tensor
            shape = [n, c', h // scale, w // scale] -- stream is None
                or
            shape = [n, c' // s, h // scale, w // scale] -- stream is not None
        """
        if self._past:
            assert self._past["stream"] == stream

            h = self._past["h"]
            c = self._past["c"]
        else:
            self._past["stream"] = stream

            N, _, H, W = inputs[0].shape
            channels = self.channels if stream is None else self.channels // self.streams
            h = c = torch.zeros(N, channels, H, W, device=self.device)

        if self.groups > 1:
            x = self.proj_x([h, *inputs], stream=stream)
        else:
            x = self.proj_x(inputs, stream=stream)

        xh = [
            to_groups_2d(x, self.groups),
            to_groups_2d(h, self.groups),
        ]
        xh = torch.stack(xh, 2)
        xh = self.drop(xh, p=dropout).flatten(1, 3)

        if stream is None:
            heads = self.groups * self.streams
            tau = torch.cat(list(self.tau))
        else:
            heads = self.groups
            tau = self.tau[stream]

        q = to_groups_2d(self.proj_q([xh], stream=stream), heads).flatten(3, 4)
        k = to_groups_2d(self.proj_k([xh], stream=stream), heads).flatten(3, 4)
        v = to_groups_2d(self.proj_v([xh], stream=stream), heads).flatten(3, 4)

        w = torch.einsum("G, N H C Q , N H C D -> N H Q D", tau, q, k).softmax(dim=3)
        a = torch.einsum("N H C D , N H Q D -> N H C Q", v, w).view_as(x)

        i = torch.sigmoid(self.proj_i([a, xh], stream=stream))
        f = torch.sigmoid(self.proj_f([a, xh], stream=stream))
        g = torch.tanh(self.proj_g([a, xh], stream=stream))
        o = torch.sigmoid(self.proj_o([a, xh], stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)

        self._past["c"] = c
        self._past["h"] = h

        return h
