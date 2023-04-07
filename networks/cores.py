import torch
from itertools import chain

from .containers import Module


class Core(Module):
    def init(self, perspectives, grids, modulations):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels, c
        grids : int
            grid channels, g
        modulations : int
            modulation features, m
        """
        raise NotImplementedError

    @property
    def channels(self):
        raise NotImplementedError

    @property
    def grid_scale(self):
        raise NotImplementedError

    def forward(self, perspective, grid, modulation, dropout=0):
        """
        Parameters
        ----------
        perspective : Tensor
            shape = [n, c, h, w]
        grid : Tensor
            shape = [g, h, w]
        modulation : Tensor
            shape = [n, m]
        dropout : float
            dropout probability

        Returns
        -------
        Tensor
            shape = [n, c', h', w']
        """
        raise NotImplementedError()


class FeedforwardRecurrent(Core):
    def __init__(self, feedforward, recurrent):
        """
        Parameters
        ----------
        feedforward : .feedforwards.Feedforward
            feedforward network
        recurrent : .recurrents.Feedforward
            recurrent network
        """
        super().__init__()
        self.feedforward = feedforward
        self.recurrent = recurrent
        self.recurrent.add_input(
            channels=self.feedforward.channels,
        )

    def init(self, perspectives, grids, modulations):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels, c
        grids : int
            grid channels, g
        modulations : int
            modulation features, m
        """
        self.feedforward.add_input(
            channels=perspectives,
        )
        self.recurrent.add_input(
            channels=grids,
        )
        self.recurrent.add_input(
            channels=modulations,
        )

    @property
    def channels(self):
        return self.recurrent.channels

    @property
    def grid_scale(self):
        return self.feedforward.scale

    def forward(self, perspective, grid, modulation, dropout=0):
        """
        Parameters
        ----------
        perspective : Tensor
            shape = [n, c, h, w]
        grid : Tensor
            shape = [g, h, w]
        modulation : Tensor
            shape = [n, m]
        dropout : float
            dropout probability

        Returns
        -------
        Tensor
            shape = [n, c', h', w']
        """
        inputs = [
            self.feedforward([perspective]),
            grid[None, :, :, :],
            modulation[:, :, None, None],
        ]
        return self.recurrent(inputs, dropout=dropout)
