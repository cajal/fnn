import torch
from collections import deque
from .modules import Module


# -------------- Core Base --------------


class Core(Module):
    """Core Module"""

    def _init(self, perspectives, modulations, streams):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels per stream (P)
        modulations : int
            modulation features per stream (M)
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
            core channels per stream (C)
        """
        raise NotImplementedError()

    def forward(self, perspective, modulation, stream=None):
        """
        Parameters
        ----------
        perspective : Tensor
            [N, S*P, H, W] -- stream is None
                or
            [N, P, H, W] -- stream is int
        modulation : Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*C, H', W'] -- stream is None
                or
            [N, C, H', W'] -- stream is int
        """
        raise NotImplementedError()


# -------------- Core Types --------------


class FeedforwardRecurrent(Core):
    """Feedforward & Recurrent Core"""

    def __init__(self, feedforward, recurrent):
        """
        Parameters
        ----------
        feedforward : fnn.model.feedforwards.Feedforward
            feedforward network
        recurrent : fnn.model.recurrents.Recurrent
            recurrent network
        """
        super().__init__()
        self.feedforward = feedforward
        self.recurrent = recurrent

    def _init(self, perspectives, modulations, streams):
        """
        Parameters
        ----------
        perspectives : int
            perspective channels (P)
        modulations : int
            modulation features per stream (M)
        streams : int
            number of streams (S)
        """
        self.perspectives = int(perspectives)
        self.modulations = int(modulations)
        self.streams = int(streams)

        self.feedforward._init(
            inputs=[self.perspectives],
            streams=self.streams,
        )
        self.recurrent._init(
            inputs=[self.feedforward.channels, self.modulations],
            streams=self.streams,
        )

    @property
    def channels(self):
        """
        Returns
        -------
        int
            core channels per stream (C)
        """
        return self.recurrent.channels

    def forward(self, perspective, modulation, stream=None):
        """
        Parameters
        ----------
        perspective : Tensor
            [N, S*P, H, W] -- stream is None
                or
            [N, P, H, W] -- stream is int
        modulation : Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*C, H', W'] -- stream is None
                or
            [N, C, H', W'] -- stream is int
        """
        f = self.feedforward([perspective], stream=stream)
        m = modulation[:, :, None, None]
        r = self.recurrent([f, m], stream=stream)
        return r


class FeedforwardRecurrentDecorr(FeedforwardRecurrent):
    """Feedforward & Recurrent Core with Decorrelation Regularization"""

    def __init__(
        self, feedforward, recurrent, decorr_length=0, decorr_weight=0, decorr_rate=0, decorr_eps=1e-5
    ):
        """
        Parameters
        ----------
        feedforward : fnn.model.feedforwards.Feedforward
            feedforward network
        recurrent : fnn.model.recurrents.Recurrent
            recurrent network
        decorr_length : int
            decorrelation length (timesteps)
        decorr_weight : float
            decorrelation weight
        decorr_rate : float
            decorrelation rate
        decorr_eps : float
            decorrelation eps
        """
        assert decorr_length >= 0
        assert decorr_weight >= 0
        assert 0 <= decorr_rate < 1
        assert decorr_eps >= 0

        super().__init__(feedforward=feedforward, recurrent=recurrent)

        self.decorr_i, self.decorr_j = torch.triu_indices(self.channels, self.channels, offset=1)
        self.decorr_length = int(decorr_length)
        self.decorr_weight = float(decorr_weight)
        self.decorr_rate = float(decorr_rate)
        self.decorr_eps = float(decorr_eps)
        self.past = deque([], maxlen=self.decorr_length)

    def _reset(self):
        self.past.clear()

    def _regularize(self):
        if self.past:
            r = torch.cat(list(self.past), 2)
            c = (r[:, self.decorr_i] * r[:, self.decorr_j]).mean(2)
            v = r.pow(2).mean(2)
            c = c / (v[:, self.decorr_i] * v[:, self.decorr_j] + self.decorr_eps).sqrt()
            yield c.abs().sum() * self.decorr_weight

    def forward(self, perspective, modulation, stream=None):
        """
        Parameters
        ----------
        perspective : Tensor
            [N, S*P, H, W] -- stream is None
                or
            [N, P, H, W] -- stream is int
        modulation : Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*C, H', W'] -- stream is None
                or
            [N, C, H', W'] -- stream is int
        """
        x = super().forward(perspective=perspective, modulation=modulation, stream=stream)

        if self.training:
            N, _, H, W = x.shape
            S = self.streams if stream is None else 1

            p = x.view(N, S, self.channels, H, W)
            p = torch.einsum("N S C H W -> S C N H W", p).flatten(2)

            if self.decorr_rate:
                p = p[:, :, torch.rand(N * H * W) > (1 - self.decorr_rate)]

            self.past.append(p)

        return x
