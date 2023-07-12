import torch
from .modules import Module
from .elements import Conv, InterGroup, Accumulate, Dropout
from .utils import to_groups_2d


# -------------- Recurrent Base --------------


class Recurrent(Module):
    """Recurrent Module"""

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
        masks : Sequence[bool]
            initial mask for each input
        streams : int
            number of streams, S
        """
        raise NotImplementedError()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            output channels per stream (O)
        """
        raise NotImplementedError()

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : Sequence[Tensor]
            [[N, I, H, W] ...] -- stream is int
                or
            [[N, S*I, H, W] ...] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, O, H, W] -- stream is int
                or
            [N, S*O, H, W] -- stream is None
        """
        raise NotImplementedError()


# -------------- Recurrent Types --------------


class Rvt(Recurrent):
    """Recurrent Vision Transformer"""

    def __init__(
        self,
        recurrent_channels,
        attention_channels,
        out_channels,
        groups=1,
        heads=1,
        spatial=3,
        dropout=0,
    ):
        """
        Parameters
        ----------
        recurrent_channels : int
            recurrent channels per stream
        attention_channels : int
            attention channels per stream
        out_channels : int
            out channels per stream
        groups : int
            groups per stream
        heads : int
            heads per stream
        spatial : int
            spatial kernel size
        dropout : float
            dropout probability -- [0, 1)
        """
        if recurrent_channels % groups != 0:
            raise ValueError("Recurrent channels must be divisible by groups")

        if attention_channels % heads != 0:
            raise ValueError("Attention channels must be divisible by heads")

        super().__init__()

        self.recurrent_channels = int(recurrent_channels)
        self.attention_channels = int(attention_channels)
        self.head_channels = int(attention_channels // heads)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.heads = int(heads)
        self.spatial = int(spatial)
        self._dropout = float(dropout)

    def _init(self, inputs, masks, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
        masks : Sequence[bool]
            initial mask for each input
        streams : int
            number of streams, S
        """
        self._inputs = list(map(int, inputs))
        self.masks = list(map(bool, masks))
        self.streams = int(streams)

        assert len(self._inputs) == len(self.masks)
        assert sum(masks) > 0

        if self.groups > 1:
            gain = (sum(masks) + 1) ** -0.5
            intergroup = InterGroup(
                channels=self.recurrent_channels,
                groups=self.groups,
                streams=self.streams,
                gain=gain,
            )
            inputs = [intergroup]
        else:
            gain = sum(masks) ** -0.5
            inputs = []

        for in_channels, mask in zip(self._inputs, self.masks):
            conv = Conv(
                in_channels=in_channels,
                out_channels=self.recurrent_channels,
                out_groups=self.groups,
                streams=self.streams,
                gain=gain * mask,
            )
            inputs.append(conv)

        self.inputs = Accumulate(inputs)

        def conv(pad, gain, bias):
            return Conv(
                in_channels=self.recurrent_channels,
                out_channels=self.attention_channels,
                in_groups=self.groups,
                out_groups=self.heads,
                streams=self.streams,
                spatial=self.spatial,
                pad=pad,
                gain=gain,
                bias=bias,
            )

        self.proj_q = Accumulate(
            [
                conv(pad="zeros", gain=(self.head_channels * 2) ** -0.5, bias=None),
                conv(pad="zeros", gain=(self.head_channels * 2) ** -0.5, bias=None),
            ]
        )
        self.proj_k = Accumulate(
            [
                conv(pad=None, gain=None, bias=None),
                conv(pad=None, gain=None, bias=None),
            ]
        )
        self.proj_v = Accumulate(
            [
                conv(pad=None, gain=2**-0.5, bias=0),
                conv(pad=None, gain=2**-0.5, bias=0),
            ]
        )

        def conv_a():
            return Conv(
                in_channels=self.attention_channels,
                out_channels=self.recurrent_channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                gain=3**-0.5,
                bias=0,
            )

        def conv_xh():
            return Conv(
                in_channels=self.recurrent_channels,
                out_channels=self.recurrent_channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                spatial=self.spatial,
                pad="zeros",
                gain=3**-0.5,
                bias=None,
            )

        self.proj_i = Accumulate([conv_a(), conv_xh(), conv_xh()])
        self.proj_f = Accumulate([conv_a(), conv_xh(), conv_xh()])
        self.proj_g = Accumulate([conv_a(), conv_xh(), conv_xh()])
        self.proj_o = Accumulate([conv_a(), conv_xh(), conv_xh()])

        self.drop = Dropout(p=self._dropout)

        self.out = Conv(
            in_channels=self.recurrent_channels,
            out_channels=self.out_channels,
            streams=self.streams,
        )

        self.past = dict()

    def _restart(self):
        self.dropout(p=self._dropout)

    def _reset(self):
        self.past.clear()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            output channels per stream (O)
        """
        return self.out_channels

    def forward(self, x, stream=None):
        """
        Parameters
        ----------
        x : Sequence[Tensor]
            [[N, I, H, W] ...] -- stream is int
                or
            [[N, S*I, H, W] ...] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, O, H, W] -- stream is int
                or
            [N, S*O, H, W] -- stream is None
        """
        if stream is None:
            channels = self.streams * self.recurrent_channels
            heads = self.streams * self.heads
        else:
            channels = self.recurrent_channels
            heads = self.heads

        if self.past:
            h = self.past["h"]
            c = self.past["c"]
        else:
            h = c = torch.zeros(1, channels, 1, 1, device=self.device)

        if self.groups > 1:
            x = self.inputs([h, *x], stream=stream)
        else:
            x = self.inputs(x, stream=stream)

        h = h.expand_as(x)
        N, _, H, W = h.shape

        q = self.proj_q([x, h], stream=stream).view(N, heads, self.head_channels, -1)
        k = self.proj_k([x, h], stream=stream).view(N, heads, self.head_channels, -1)
        v = self.proj_v([x, h], stream=stream).view(N, heads, self.head_channels, -1)

        w = torch.einsum("N G C Q , N G C D -> N G Q D", q, k).softmax(dim=3)
        a = torch.einsum("N G C D , N G Q D -> N G C Q", v, w).view(N, -1, H, W)

        i = torch.sigmoid(self.proj_i([a, x, h], stream=stream))
        f = torch.sigmoid(self.proj_f([a, x, h], stream=stream))
        g = torch.tanh(self.proj_g([a, x, h], stream=stream))
        o = torch.sigmoid(self.proj_o([a, x, h], stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)
        h = self.drop(h)

        self.past["c"] = c
        self.past["h"] = h

        return self.out(h, stream=stream)


class ConvLstm(Recurrent):
    """Convolutional Lstm"""

    def __init__(
        self,
        recurrent_channels,
        out_channels,
        groups=1,
        spatial=3,
        dropout=0,
    ):
        """
        Parameters
        ----------
        recurrent_channels : int
            recurrent channels per stream
        out_channels : int
            out channels per stream
        groups : int
            groups per stream
        spatial : int
            spatial kernel size
        dropout : float
            dropout probability -- [0, 1)
        """
        if recurrent_channels % groups != 0:
            raise ValueError("Recurrent channels must be divisible by groups")

        super().__init__()

        self.recurrent_channels = int(recurrent_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.spatial = int(spatial)
        self._dropout = float(dropout)

    def _init(self, inputs, masks, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
        masks : Sequence[bool]
            initial mask for each input
        streams : int
            number of streams, S
        """
        self._inputs = list(map(int, inputs))
        self.masks = list(map(bool, masks))
        self.streams = int(streams)

        assert len(self._inputs) == len(self.masks)
        assert sum(masks) > 0

        if self.groups > 1:
            gain = (sum(masks) + 1) ** -0.5
            intergroup = InterGroup(
                channels=self.recurrent_channels,
                groups=self.groups,
                streams=self.streams,
                gain=gain,
            )
            inputs = [intergroup]
        else:
            gain = sum(masks) ** -0.5
            inputs = []

        for in_channels, mask in zip(self._inputs, self.masks):
            conv = Conv(
                in_channels=in_channels,
                out_channels=self.recurrent_channels,
                out_groups=self.groups,
                streams=self.streams,
                gain=gain * mask,
            )
            inputs.append(conv)

        self.inputs = Accumulate(inputs)

        def conv_xh():
            return Conv(
                in_channels=self.recurrent_channels,
                out_channels=self.recurrent_channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                spatial=self.spatial,
                pad="zeros",
                gain=2**-0.5,
                bias=None,
            )

        self.proj_i = Accumulate([conv_xh(), conv_xh()])
        self.proj_f = Accumulate([conv_xh(), conv_xh()])
        self.proj_g = Accumulate([conv_xh(), conv_xh()])
        self.proj_o = Accumulate([conv_xh(), conv_xh()])

        self.drop = Dropout(p=self._dropout)

        self.out = Conv(
            in_channels=self.recurrent_channels,
            out_channels=self.out_channels,
            streams=self.streams,
        )

        self.past = dict()

    def _restart(self):
        self.dropout(p=self._dropout)

    def _reset(self):
        self.past.clear()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            output channels per stream (O)
        """
        return self.out_channels

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        x : Sequence[Tensor]
            [[N, I, H, W] ...] -- stream is int
                or
            [[N, S*I, H, W] ...] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, O, H, W] -- stream is int
                or
            [N, S*O, H, W] -- stream is None
        """
        if self.past:
            h = self.past["h"]
            c = self.past["c"]
        else:
            if stream is None:
                channels = self.streams * self.recurrent_channels
            else:
                channels = self.recurrent_channels
            h = c = torch.zeros(1, channels, 1, 1, device=self.device)

        if self.groups > 1:
            x = self.inputs([h, *x], stream=stream)
        else:
            x = self.inputs(x, stream=stream)

        h = h.expand_as(x)

        i = torch.sigmoid(self.proj_i([x, h], stream=stream))
        f = torch.sigmoid(self.proj_f([x, h], stream=stream))
        g = torch.tanh(self.proj_g([x, h], stream=stream))
        o = torch.sigmoid(self.proj_o([x, h], stream=stream))

        c = f * c + i * g
        h = o * torch.tanh(c)
        h = self.drop(h)

        self.past["c"] = c
        self.past["h"] = h

        return self.out(h, stream=stream)
