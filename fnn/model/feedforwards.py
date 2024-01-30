import torch
from .modules import Module, ModuleList
from .elements import Conv, Dropout, nonlinearity
from .utils import cat_groups_2d


# -------------- Feedforward Base --------------


class Feedforward(Module):
    """Feature Module"""

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        channels : Sequence[int]
            [input channels per stream (I), ...]
        streams : int
            number of streams (S)
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

    def forward(self, inputs, stream=None):
        """
        Parameters
        ----------
        inputs : Sequence[Tensor]
            [[N, I, H, W] ...] -- stream is int
                or
            [[N, S*I, H, W] ...] -- stream is None
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, O, H//D, W//D] -- stream is int
                or
            [N, S*O, H//D, W//D] -- stream is None
        """
        raise NotImplementedError()


# -------------- Feedforward Types --------------


class Block(Module):
    """Dense Connection Block"""

    def __init__(self, channels, groups, layers, temporal, spatial, pool, nonlinear=None, dropout=0):
        """
        Parameters
        ----------
        channels : int
            channels per stream (C)
        groups : int
            groups per stream
        layers : int
            number of layers
        temporal : int
            temporal kernel size
        spatial : int
            spatial kernel size
        pool : int
            spatial pooling size
        nonlinear : str | None
            nonlinearity
        dropout : float
            dropout probability -- [0, 1)
        """
        super().__init__()
        self.channels = int(channels)
        self.groups = int(groups)
        self._layers = int(layers)
        self.temporal = int(temporal)
        self.spatial = int(spatial)
        self.pool = int(pool)
        self.nonlinear, self.gamma = nonlinearity(nonlinear)
        self._dropout = float(dropout)

    def _init(self, streams):
        """
        Parameters
        ----------
        streams : int
            number of streams (S)
        """
        self.streams = int(streams)

        def drop(_):
            return Dropout(p=self._dropout)

        def conv(layer):
            return Conv(
                in_channels=self.channels,
                out_channels=self.channels,
                in_groups=self.groups,
                out_groups=self.groups,
                streams=self.streams,
                temporal=self.temporal,
                spatial=self.spatial,
                gain=2**-0.5 if layer else 1,
            )

        def skip(layer):
            if layer:
                return Conv(
                    in_channels=self.channels * layer,
                    out_channels=self.channels,
                    in_groups=self.groups,
                    out_groups=self.groups,
                    streams=self.streams,
                    gain=2**-0.5,
                    bias=None,
                )

        self.drops = ModuleList(map(drop, range(self._layers)))
        self.convs = ModuleList(map(conv, range(self._layers)))
        self.skips = ModuleList(map(skip, range(self._layers)))

        if self.pool == 1:
            self.pool_fn = lambda x: x
        else:
            self.pool_fn = lambda x: torch.nn.functional.avg_pool2d(x, self.pool)

    def _restart(self):
        self.dropout(p=self._dropout)

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
        4D Tensor
            [N, C, H', W'] -- stream is int
                or
            [N, S*C, H', W'] -- stream is None
        """
        if stream is None:
            groups = self.groups * self.streams
        else:
            groups = self.groups

        for l, (drop, conv, skip) in enumerate(zip(self.drops, self.convs, self.skips)):
            y = self.nonlinear(x) * self.gamma
            y = drop(y)

            if skip is None:
                z = x
                x = conv(y, stream=stream)
                n = y
            else:
                x = conv(y, stream=stream) + skip(n, stream=stream)
                if l + 1 < self._layers:
                    n = cat_groups_2d([n, y], groups=groups)

            z = cat_groups_2d([z, x], groups=groups)

        return self.pool_fn(z)


class Dense(Feedforward):
    """Dense Connection Network"""

    def __init__(
        self,
        in_spatial,
        in_stride,
        block_channels,
        block_groups,
        block_layers,
        block_temporals,
        block_spatials,
        block_pools,
        out_channels,
        nonlinear=None,
        dropout=0,
    ):
        """
        Parameters
        ----------
        in_spatial : int
            input spatial kernel size
        in_stride : int
            input spatial stride
        block_channels : Sequence[int]
            block channels per stream
        block_groups : Sequence[int]
            block groups per stream
        block_layers : Sequence[int]
            block layers
        block_temporals : Sequence[int]
            block temporal kernel sizes
        block_spatials : Sequence[int]
            block spatial kernel sizes
        block_pools : Sequence[int]
            block spatial pooling sizes
        out_channels : int
            output channels per strem
        nonlinear : str | None
            nonlinearity
        dropout : float
            dropout probability -- [0, 1)
        """
        assert (
            len(block_channels)
            == len(block_groups)
            == len(block_layers)
            == len(block_temporals)
            == len(block_spatials)
            == len(block_pools)
        )
        super().__init__()

        self.in_spatial = int(in_spatial)
        self.in_stride = int(in_stride)
        self.block_channels = list(map(int, block_channels))
        self.block_groups = list(map(int, block_groups))
        self.block_layers = list(map(int, block_layers))
        self.block_temporals = list(map(int, block_temporals))
        self.block_spatials = list(map(int, block_spatials))
        self.block_pools = list(map(int, block_pools))
        self.out_channels = int(out_channels)
        self.nonlinear = str(nonlinear)
        self._dropout = float(dropout)

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        inputs : Sequence[int]
            [input channels per stream (I), ...]
        streams : int
            number of streams (S)
        """
        self._inputs = list(map(int, inputs))
        self.streams = int(streams)

        def proj(in_channels, out_channels, out_groups, spatial, stride):
            return Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                out_groups=out_groups,
                spatial=spatial,
                stride=stride,
                pad="replicate",
                streams=self.streams,
            )

        self.inputs = ModuleList()
        self.blocks = ModuleList()

        in_channels = sum(self._inputs)
        in_spatial = self.in_spatial
        in_stride = self.in_stride

        for channels, groups, layers, temporal, spatial, pool in zip(
            self.block_channels,
            self.block_groups,
            self.block_layers,
            self.block_temporals,
            self.block_spatials,
            self.block_pools,
        ):
            inputs = proj(
                in_channels=in_channels,
                out_channels=channels,
                out_groups=groups,
                spatial=in_spatial,
                stride=in_stride,
            )
            self.inputs.append(inputs)

            block = Block(
                channels=channels,
                groups=groups,
                layers=layers,
                temporal=temporal,
                spatial=spatial,
                pool=pool,
                nonlinear=self.nonlinear,
                dropout=self._dropout,
            )
            block._init(streams=self.streams)
            self.blocks.append(block)

            in_channels = (layers + 1) * channels
            in_spatial = 1
            in_stride = 1

        self.out = proj(
            in_channels=in_channels,
            out_channels=self.out_channels,
            out_groups=1,
            spatial=1,
            stride=1,
        )

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
            [N, O, H', W'] -- stream is int
                or
            [N, S*O, H', W'] -- stream is None
        """
        if stream is None:
            x = cat_groups_2d(x, groups=self.streams)
        else:
            x = cat_groups_2d(x, groups=1)

        for inp, block in zip(self.inputs, self.blocks):
            x = inp(x, stream=stream)
            x = block(x, stream=stream)

        return self.out(x, stream=stream)


class InputDense(Feedforward):
    """Input & Dense Connection Network"""

    def __init__(
        self,
        input_spatial,
        input_stride,
        block_channels,
        block_groups,
        block_layers,
        block_temporals,
        block_spatials,
        block_pools,
        out_channels,
        nonlinear=None,
        dropout=0,
    ):
        """
        Parameters
        ----------
        input_spatial : int
            input spatial kernel size
        input_stride : int
            input spatial stride
        block_channels : Sequence[int]
            block channels per stream
        block_groups : Sequence[int]
            block groups per stream
        block_layers : Sequence[int]
            block layers
        block_temporals : Sequence[int]
            block temporal kernel sizes
        block_spatials : Sequence[int]
            block spatial kernel sizes
        block_pools : Sequence[int]
            block spatial pooling sizes
        out_channels : int
            output channels per strem
        nonlinear : str | None
            nonlinearity
        dropout : float
            dropout probability -- [0, 1)
        """
        assert (
            len(block_channels)
            == len(block_groups)
            == len(block_layers)
            == len(block_temporals)
            == len(block_spatials)
            == len(block_pools)
        )
        super().__init__()

        self.input_spatial = int(input_spatial)
        self.input_stride = int(input_stride)
        self.block_channels = list(map(int, block_channels))
        self.block_groups = list(map(int, block_groups))
        self.block_layers = list(map(int, block_layers))
        self.block_temporals = list(map(int, block_temporals))
        self.block_spatials = list(map(int, block_spatials))
        self.block_pools = list(map(int, block_pools))
        self.out_channels = int(out_channels)
        self.nonlinear = str(nonlinear)
        self._dropout = float(dropout)

    def _init(self, inputs, streams):
        """
        Parameters
        ----------
        inputs : Sequence[int]
            [input channels per stream (I), ...]
        streams : int
            number of streams (S)
        """
        self._inputs = list(map(int, inputs))
        self.streams = int(streams)

        def proj(in_channels, out_channels, out_groups, spatial, stride, wnorm):
            return Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                out_groups=out_groups,
                spatial=spatial,
                stride=stride,
                wnorm=wnorm,
                pad="replicate",
                streams=self.streams,
            )

        self.inputs = ModuleList()
        self.blocks = ModuleList()

        in_channels = sum(self._inputs)
        in_spatial = self.input_spatial
        in_stride = self.input_stride
        in_wnorm = False

        for channels, groups, layers, temporal, spatial, pool in zip(
            self.block_channels,
            self.block_groups,
            self.block_layers,
            self.block_temporals,
            self.block_spatials,
            self.block_pools,
        ):
            inputs = proj(
                in_channels=in_channels,
                out_channels=channels,
                out_groups=groups,
                spatial=in_spatial,
                stride=in_stride,
                wnorm=in_wnorm,
            )
            self.inputs.append(inputs)

            block = Block(
                channels=channels,
                groups=groups,
                layers=layers,
                temporal=temporal,
                spatial=spatial,
                pool=pool,
                nonlinear=self.nonlinear,
                dropout=self._dropout,
            )
            block._init(streams=self.streams)
            self.blocks.append(block)

            in_channels = (layers + 1) * channels
            in_spatial = 1
            in_stride = 1
            in_wnorm = True

        self.out = proj(
            in_channels=in_channels,
            out_channels=self.out_channels,
            out_groups=1,
            spatial=1,
            stride=1,
            wnorm=True,
        )

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
            [N, O, H', W'] -- stream is int
                or
            [N, S*O, H', W'] -- stream is None
        """
        if stream is None:
            x = cat_groups_2d(x, groups=self.streams)
        else:
            x = cat_groups_2d(x, groups=1)

        for inp, block in zip(self.inputs, self.blocks):
            x = inp(x, stream=stream)
            x = block(x, stream=stream)

        return self.out(x, stream=stream)
