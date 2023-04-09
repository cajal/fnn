import torch
from torch.nn.functional import grid_sample

from .containers import Module
from .elements import Dropout, Conv


class Readout(Module):
    @property
    def streams(self):
        raise NotImplementedError()

    def init(self, cores, units):
        """
        Parameters
        ----------
        cores : int
            core channels, c
        units : int
            response units, u
        """
        raise NotImplementedError()

    def forward(self, core, dropout=0):
        """
        Parameters
        ----------
        core : Tensor
            shape = [n, c, h, w]
        dropout : float
            dropout probability

        Returns
        -------
        Tensor
            shape = [n, u]
        """
        raise NotImplementedError()


class PositionFeatures(Readout):
    def __init__(self, channels, position, bound, features):
        """
        Parameters
        ----------
        channels : int
            readout channels
        position : .positions.Position
            spatial position
        bounds : .bounds.Bound
            spatial bound
        features : .features.Features
            feature weights
        """
        assert bound.vmin == -1 and bound.vmax == 1
        super().__init__()

        self.channels = int(channels)
        self.drop = Dropout(
            drop_dim=[2, 3],
            reduce_dim=[1],
        )
        self.proj = Conv(
            out_channels=self.channels,
        )
        self.position = position
        self.bound = bound
        self.features = features

    def init(self, cores, units):
        """
        Parameters
        ----------
        cores : int
            core channels, c
        units : int
            response units, u
        """
        self.proj.add_input(
            in_channels=cores,
        )
        self.position.init(
            units=units,
        )
        self.features.init(
            units=units,
            features=self.channels,
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(units),
        )

    def _param_groups(self, **kwargs):
        if kwargs.get("weight_decay"):
            kwargs.update(weight_decay=0)
            yield dict(params=[self.bias], **kwargs)
