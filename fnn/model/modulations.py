import torch
from torch.nn import init
from .modules import Module
from .elements import Linear, FlatDropout, Mlp, Lstm
from .parameters import Parameter, ParameterList
from .utils import cat_groups


# -------------- Modulation Base --------------


class Modulation(Module):
    """Modulation Model"""

    def _init(self, modulations, streams):
        """
        Parameters
        ----------
        modulations : int
            modulation inputs per stream (I)
        streams : int
            number of streams (S)
        """
        raise NotImplementedError()

    @property
    def features(self):
        """
        Returns
        -------
        int
            modulation features per stream (M)
        """
        raise NotImplementedError()

    def forward(self, modulation, stream=None):
        """
        Parameters
        ----------
        modulation : Tensor
            [N, S*I] -- stream is None
                or
            [N, I] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        """
        raise NotImplementedError()


# -------------- Modulation Types --------------


class FlatLstm(Modulation):
    """Flat Lstm"""

    def __init__(self, in_features, out_features, hidden_features, init_input=-1, init_forget=1, dropout=0):
        """
        Parameters
        ----------
        in_features : int
            in features per stream
        out_features : int
            out features per stream
        hidden_features : int
            hidden features per stream
        init_input : float
            initial input gate bias
        init_forget : float
            initial forget gate bias
        dropout : float
            dropout probability -- [0, 1)
        """
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden_features = int(hidden_features)

        self.init_input = float(init_input)
        self.init_forget = float(init_forget)

        self._dropout = float(dropout)

    def _init(self, modulations, streams):
        """
        Parameters
        ----------
        modulations : int
            modulation inputs per stream (I)
        streams : int
            number of streams (S)
        """
        self.modulations = int(modulations)
        self.streams = int(streams)

        self.drop_x = FlatDropout(p=self._dropout)
        self.drop_h = FlatDropout(p=self._dropout)

        self.proj_x = Linear(
            in_features=self.modulations,
            out_features=self.in_features,
            streams=self.streams,
        )

        def linear(bias):
            return Linear(
                in_features=self.in_features + self.hidden_features,
                out_features=self.hidden_features,
                streams=self.streams,
                bias=bias,
            )

        self.proj_i = linear(bias=self.init_input)
        self.proj_f = linear(bias=self.init_forget)
        self.proj_g = linear(bias=0)
        self.proj_o = linear(bias=0)

        self.out = Linear(
            in_features=self.hidden_features,
            out_features=self.out_features,
            streams=self.streams,
            gain=0,
        )
        self.past = dict()

    def _restart(self):
        self.dropout(p=self._dropout)

    def _reset(self):
        self.past.clear()

    @property
    def features(self):
        """
        Returns
        -------
        int
            modulation features per stream (M)
        """
        return self.out_features

    def forward(self, modulation, stream=None):
        """
        Parameters
        ----------
        modulation : Tensor
            [N, S*I] -- stream is None
                or
            [N, I] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        """
        if stream is None:
            S = self.streams
        else:
            S = 1

        if self.past:
            h = self.past["h"]
            c = self.past["c"]
        else:
            h = c = torch.zeros([1, S * self.hidden_features], device=self.device)

        x = torch.tanh(self.proj_x(modulation, stream=stream))
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

        return self.out(h, stream=stream)


class Sigmoid(Module):
    """Sigmoid"""

    def _init(self, modulations, streams):
        """
        Parameters
        ----------
        modulations : int
            modulation inputs per stream (I)
        streams : int
            number of streams (S)
        """
        self.modulations = int(modulations)
        self.streams = int(streams)

        def param(value):
            return Parameter(torch.full([1, self.modulations], value))

        self.gains = ParameterList([param(1.0) for _ in range(self.streams)])
        self.gains.decay = False
        self.gains.norm_dim = 0

        self.biases = ParameterList([param(0.0) for _ in range(self.streams)])
        self.biases.decay = False
        self.biases.norm_dim = 0

        self.past = dict()

    def _reset(self):
        self.past.clear()

    def gain(self, stream=None):
        """
        Parameters
        ----------
        stream : int | None
            specific stream (int) or all streams (None)
        """
        if stream is None:
            return torch.cat(list(self.gains), dim=1)
        else:
            return self.gains[stream]

    def bias(self, stream=None):
        """
        Parameters
        ----------
        stream : int | None
            specific stream (int) or all streams (None)
        """
        if stream is None:
            return torch.cat(list(self.biases), dim=1)
        else:
            return self.biases[stream]

    def forward(self, modulation, stream=None):
        """
        Parameters
        ----------
        modulation : Tensor
            [N, S*I] -- stream is None
                or
            [N, I] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*I] -- stream is None
                or
            [N, I] -- stream is int
        """
        if self.past:
            assert self.past["stream"] == stream
            gain = self.past["gain"]
            bias = self.past["bias"]

        else:
            self.past["stream"] = stream
            self.past["gain"] = gain = self.gain(stream)
            self.past["bias"] = bias = self.bias(stream)

        return torch.sigmoid(modulation * gain + bias)


class SigmoidLstm(Modulation):
    """Sigmoid Lstm"""

    def __init__(self, in_features, out_features, hidden_features, init_input=-1, init_forget=1, dropout=0):
        """
        Parameters
        ----------
        in_features : int
            in features per stream
        out_features : int
            out features per stream
        hidden_features : int
            hidden features per stream
        init_input : float
            initial input gate bias
        init_forget : float
            initial forget gate bias
        dropout : float
            dropout probability -- [0, 1)
        """
        super().__init__()

        self.sigmoid = Sigmoid()

        self.lstm = FlatLstm(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            init_input=init_input,
            init_forget=init_forget,
            dropout=dropout,
        )

    def _init(self, modulations, streams):
        """
        Parameters
        ----------
        modulations : int
            modulation inputs per stream (I)
        streams : int
            number of streams (S)
        """
        self.sigmoid._init(
            modulations=modulations,
            streams=streams,
        )
        self.lstm._init(
            modulations=modulations,
            streams=streams,
        )

    @property
    def features(self):
        """
        Returns
        -------
        int
            modulation features per stream (M)
        """
        return self.lstm.out_features

    def forward(self, modulation, stream=None):
        """
        Parameters
        ----------
        modulation : Tensor
            [N, S*I] -- stream is None
                or
            [N, I] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        """
        modulation = self.sigmoid(modulation, stream=stream)
        modulation = self.lstm(modulation, stream=stream)
        return modulation


class MlpLstm(Modulation):
    """Mlp Lstm"""

    def __init__(
        self,
        mlp_features,
        mlp_layers,
        mlp_nonlinear,
        lstm_features,
        init_input=-1,
        init_forget=1,
        dropout=0,
    ):
        """
        Parameters
        ----------
        mlp_features : int
            mlp features per stream
        mlp_layers : int
            mlp layers
        mlp_nonlinear : str
            mlp nonlinearity
        lstm_features : int
            lstm features per stream
        init_input : float
            initial input gate bias
        init_forget : float
            initial forget gate bias
        dropout : float
            dropout probability -- [0, 1)
        """
        super().__init__()

        self.mlp_features = int(mlp_features)
        self.mlp_layers = int(mlp_layers)
        self.mlp_nonlinear = str(mlp_nonlinear)
        self.lstm_features = int(lstm_features)
        self.init_input = float(init_input)
        self.init_forget = float(init_forget)
        self._dropout = float(dropout)

    def _init(self, modulations, streams):
        """
        Parameters
        ----------
        modulations : int
            modulation inputs per stream (I)
        streams : int
            number of streams (S)
        """
        self.modulations = int(modulations)
        self.streams = int(streams)

        self.mlp = Mlp(
            in_features=self.modulations,
            out_features=[self.mlp_features] * self.mlp_layers,
            out_wnorms=[False] + [True] * (self.mlp_layers - 1),
            out_nonlinears=[self.mlp_nonlinear] * self.mlp_layers,
            streams=self.streams,
        )

        for weight in self.mlp.linears[0].weights:
            init.zeros_(weight)

        self.lstm = Lstm(
            in_features=self.mlp_features,
            out_features=self.lstm_features,
            streams=self.streams,
            dropout=self._dropout,
            init_input=self.init_input,
            init_forget=self.init_forget,
        )

        self.past = dict()

    def _reset(self):
        self.past.clear()

    @property
    def features(self):
        """
        Returns
        -------
        int
            modulation features per stream (M)
        """
        return self.lstm_features

    def forward(self, modulation, stream=None):
        """
        Parameters
        ----------
        modulation : Tensor
            [N, S*I] -- stream is None
                or
            [N, I] -- stream is int
        stream : int | None
            specific stream (int) or all streams (None)

        Returns
        -------
        Tensor
            [N, S*M] -- stream is None
                or
            [N, M] -- stream is int
        """
        modulation = self.mlp(modulation, stream=stream)
        modulation = self.lstm(modulation, stream=stream)
        return modulation
