import torch
from .modules import Module
from .parameters import Parameter, ParameterList


# -------------- Feature Base --------------


class Feature(Module):
    """Feature Module"""

    def _init(self, inputs, outputs, units, streams):
        """
        Parameters
        ----------
        inputs : int
            inputs per stream (I)
        outputs : int
            outputs per unit and stream (O)
        units : int
            number of units (U)
        streams : int
            number of streams (S)
        """
        raise NotImplementedError()

    def forward(self, stream=None):
        """
        Parameters
        ----------
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [U, O, I] -- stream is int
                or
            [S, U, O, I] -- stream is None
        """
        raise NotImplementedError()


# -------------- Feature Types --------------


class Norm(Feature):
    """Norm Feature"""

    def __init__(self, groups=1, eps=1e-5):
        """
        Parameters
        ----------
        groups : int
            groups per stream
        eps : float
            small value added to denominator for numerical stability
        """
        super().__init__()

        self.groups = int(groups)
        self.eps = float(eps)

        self.past = dict()

    def _init(self, inputs, outputs, units, streams):
        """
        Parameters
        ----------
        inputs : int
            inputs per stream (I)
        outputs : int
            outputs per unit and stream (O)
        units : int
            number of units (U)
        streams : int
            number of streams (S)
        """
        self.inputs = int(inputs)
        self.outputs = int(outputs)
        self.units = int(units)
        self.streams = int(streams)
        self.fan_in = self.inputs // self.groups

        if self.fan_in * self.groups != self.inputs:
            raise ValueError("Inputs must be divisible by groups")

        weight = lambda: Parameter(torch.zeros([self.units, self.outputs, self.groups, self.fan_in]))
        self.weights = ParameterList([weight() for _ in range(self.streams)])
        self.weights.scale = self.units
        self.weights.norm_dim = 3

        gain = lambda: Parameter(torch.full([self.units, self.outputs, self.groups], self.groups**-0.5))
        self.gains = ParameterList([gain() for _ in range(self.streams)])
        self.gains.scale = self.units
        self.gains.decay = False
        self.gains.norm_dim = 0

    def _reset(self):
        self.past.clear()

    def weight(self, stream):
        """
        Parameters
        ----------
        stream : int
            stream index

        Returns
        -------
        Tensor
            [U, O, I]
        """
        if stream is None:
            weights = [self.weight(stream=s) for s in range(self.streams)]
            return torch.stack(weights, dim=0)

        else:
            weight = self.weights[stream]
            gain = self.gains[stream]

            var, mean = torch.var_mean(weight, dim=3, unbiased=False, keepdim=True)
            scale = (var * self.inputs + self.eps).pow(-0.5)

            weight = (weight - mean) * scale
            return torch.einsum("U O G I, U O G -> U O G I", weight, gain).flatten(start_dim=2)

    def forward(self, stream=None):
        """
        Parameters
        ----------
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [U, O, I] -- stream is int
                or
            [S, U, O, I] -- stream is None
        """
        if self.past:
            assert self.past["stream"] == stream
            weight = self.past["weight"]

        else:
            self.past["stream"] = stream
            self.past["weight"] = weight = self.weight(stream)

        return weight


class Vanilla(Feature):
    """Vanilla Feature"""

    def _init(self, inputs, outputs, units, streams):
        """
        Parameters
        ----------
        inputs : int
            inputs per stream (I)
        outputs : int
            outputs per unit and stream (O)
        units : int
            number of units (U)
        streams : int
            number of streams (S)
        """
        self.inputs = int(inputs)
        self.outputs = int(outputs)
        self.units = int(units)
        self.streams = int(streams)

        weight = lambda: Parameter(torch.zeros([self.units, self.outputs, self.inputs]))
        self.weights = ParameterList([weight() for _ in range(self.streams)])
        self.weights.scale = self.units
        self.weights.norm_dim = 2

    def forward(self, stream=None):
        """
        Parameters
        ----------
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [U, O, I] -- stream is int
                or
            [S, U, O, I] -- stream is None
        """
        if stream is None:
            return torch.stack(list(self.weights), dim=0)

        else:
            return self.weights[stream]
