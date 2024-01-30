from functools import reduce
from torch.nn import init
from .modules import Module, ModuleList
from .elements import Linear, FlatDropout, Mlp, nonlinearity
from .utils import isotropic_grid_sample_2d, rmat_3d


# -------------- Perspective Base --------------


class Perspective(Module):
    """Perspective Module"""

    def _init(self, stimuli, perspectives):
        """
        Parameters
        ----------
        stimuli : int
            stimulus channels (S)
        perspectives : int
            perspective features (F)
        """
        raise NotImplementedError()

    @property
    def channels(self):
        """
        Returns
        -------
        int
            perspective channels (P)
        """
        raise NotImplementedError()

    def forward(self, stimulus, perspective, pad_mode="zeros"):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, S, H, W]
        perspective : Tensor
            [N, F]
        pad_mode : str
            "zeros" | "replicate"

        Returns
        -------
        Tensor
            [N, P, H', W']
        """
        raise NotImplementedError()

    def inverse(self, stimulus, perspective, height=144, width=256, pad_mode="zeros"):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, S, H, W]
        perspective : Tensor
            [N, F]
        height : int
            output height (H')
        width : int
            output width (W')
        pad_mode : str
            "zeros" | "replicate"

        Returns
        -------
        Tensor
            [N, P, H', W']
        """
        raise NotImplementedError()


# -------------- Perspective Types --------------


class MonitorRetina(Perspective):
    """Monitor & Retina Perspective"""

    def __init__(
        self,
        monitor,
        monitor_pixel,
        retina,
        retina_pixel,
        height,
        width,
        features,
        nonlinear=None,
        dropout=0,
    ):
        """
        Parameters
        ----------
        monitor : fnn.model.monitors.Monitor
            3D monitor model
        monitor_pixel : fnn.model.pixels.Pixel
            monitor pixel intensity
        retina : fnn.model.retinas.Retina
            3D retina model
        retina_pixel : fnn.model.pixels.Pixel
            retina pixel intensity
        height : int
            retina height
        width : int
            retina width
        features : Sequence[int]
            mlp features
        nonlinear : str | None
            nonlinearity
        dropout : float
            dropout probability -- [0, 1)
        """
        super().__init__()

        self.monitor = monitor
        self.monitor_pixel = monitor_pixel

        self.retina = retina
        self.retina._init(height=height, width=width)
        self.retina_pixel = retina_pixel

        self.features = list(map(int, features))
        self.nonlinear, self.gamma = nonlinearity(nonlinear=nonlinear)
        self._dropout = float(dropout)

    def _init(self, stimuli, perspectives):
        """
        Parameters
        ----------
        stimuli : int
            stimulus channels (S)
        perspectives : int
            perspective features (F)
        """
        self._channels = int(stimuli)
        self.perspectives = int(perspectives)

        self.layers = ModuleList([])
        in_features = perspectives
        for out_features in self.features:
            linear = Linear(in_features=in_features, out_features=out_features)
            in_features = out_features
            self.layers.append(linear)

        self.drop = FlatDropout(p=self._dropout)

        self.out = Linear(in_features=in_features, out_features=3, gain=0)

    def _restart(self):
        self.dropout(p=self._dropout)

    @property
    def channels(self):
        """
        Returns
        -------
        int
            perspective channels (P)
        """
        return self._channels

    def rmat(self, perspective):
        """
        Parameters
        ----------
        perspective : Tensor
            [N, F]

        Returns
        -------
        Tensor
            [N, 3, 3], 3D rotation matrix
        """
        x = reduce(lambda x, layer: self.nonlinear(layer(x)) * self.gamma, self.layers, perspective)
        x = self.drop(x)
        x = self.out(x)
        return rmat_3d(*x.unbind(1))

    def forward(self, stimulus, perspective, pad_mode="zeros"):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, S, H, W]
        perspective : Tensor
            [N, F]
        pad_mode : str
            "zeros" | "replicate"

        Returns
        -------
        Tensor
            [N, P, H', W']
        """
        size = max(stimulus.size(0), perspective.size(0))

        rmat = self.rmat(perspective).expand(size, -1, -1)
        rays = self.retina.rays(rmat)
        grid = self.monitor.project(rays)

        pixels = self.monitor_pixel(stimulus).expand(size, -1, -1, -1)
        pixels = isotropic_grid_sample_2d(pixels, grid=grid, pad_mode=pad_mode)
        pixels = self.retina_pixel(pixels)

        return pixels

    def inverse(self, stimulus, perspective, height=144, width=256, pad_mode="zeros"):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, S, H, W]
        perspective : Tensor
            [N, F]
        height : int
            output height (H')
        width : int
            output width (W')
        pad_mode : str
            "zeros" | "replicate"

        Returns
        -------
        Tensor
            [N, P, H', W']
        """
        size = max(stimulus.size(0), perspective.size(0))

        rmat = self.rmat(perspective).expand(size, -1, -1)
        rays = self.monitor.rays(size, height, width)
        grid = self.retina.project(rays, rmat)

        pixels = self.retina_pixel.inverse(stimulus).expand(size, -1, -1, -1)
        pixels = isotropic_grid_sample_2d(pixels, grid=grid, pad_mode=pad_mode)
        pixels = self.monitor_pixel.inverse(pixels)

        return pixels


class MlpMonitorRetina(Perspective):
    """Mlp, Monitor, Retina Perspective"""

    def __init__(
        self,
        mlp_features,
        mlp_layers,
        mlp_nonlinear,
        monitor,
        monitor_pixel,
        retina,
        retina_pixel,
        height,
        width,
    ):
        """
        Parameters
        ----------
        mlp_features : int
            mlp features
        mlp_layers : int
            mlp layers
        mlp_nonlinear : str
            mlp nonlinearity
        monitor : fnn.model.monitors.Monitor
            3D monitor model
        monitor_pixel : fnn.model.pixels.Pixel
            monitor pixel intensity
        retina : fnn.model.retinas.Retina
            3D retina model
        retina_pixel : fnn.model.pixels.Pixel
            retina pixel intensity
        height : int
            retina height
        width : int
            retina width
        """
        super().__init__()

        self.mlp_features = int(mlp_features)
        self.mlp_layers = int(mlp_layers)
        self.mlp_nonlinear = str(mlp_nonlinear)

        self.monitor = monitor
        self.monitor_pixel = monitor_pixel

        self.retina = retina
        self.retina_pixel = retina_pixel

        self.height = int(height)
        self.width = int(width)

    def _init(self, stimuli, perspectives):
        """
        Parameters
        ----------
        stimuli : int
            stimulus channels (S)
        perspectives : int
            perspective features (F)
        """
        self._channels = int(stimuli)
        self.perspectives = int(perspectives)

        self.retina._init(
            height=self.height,
            width=self.width,
        )
        self.mlp = Mlp(
            in_features=self.perspectives,
            out_features=[self.mlp_features] * self.mlp_layers + [3],
            out_wnorms=[False] + [True] * self.mlp_layers,
            out_nonlinears=[self.mlp_nonlinear] * self.mlp_layers + [None],
        )
        init.zeros_(self.mlp.linears[0].weights[0])

    @property
    def channels(self):
        """
        Returns
        -------
        int
            perspective channels (P)
        """
        return self._channels

    def rmat(self, perspective):
        """
        Parameters
        ----------
        perspective : Tensor
            [N, F]

        Returns
        -------
        Tensor
            [N, 3, 3], 3D rotation matrix
        """
        return rmat_3d(*self.mlp(perspective).unbind(1))

    def forward(self, stimulus, perspective, pad_mode="zeros"):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, S, H, W]
        perspective : Tensor
            [N, F]
        pad_mode : str
            "zeros" | "replicate"

        Returns
        -------
        Tensor
            [N, P, H', W']
        """
        size = max(stimulus.size(0), perspective.size(0))

        rmat = self.rmat(perspective).expand(size, -1, -1)
        rays = self.retina.rays(rmat)
        grid = self.monitor.project(rays)

        pixels = self.monitor_pixel(stimulus).expand(size, -1, -1, -1)
        pixels = isotropic_grid_sample_2d(pixels, grid=grid, pad_mode=pad_mode)
        pixels = self.retina_pixel(pixels)

        return pixels

    def inverse(self, stimulus, perspective, height=144, width=256, pad_mode="zeros"):
        """
        Parameters
        ----------
        stimulus : Tensor
            [N, S, H, W]
        perspective : Tensor
            [N, F]
        height : int
            output height (H')
        width : int
            output width (W')
        pad_mode : str
            "zeros" | "replicate"

        Returns
        -------
        Tensor
            [N, P, H', W']
        """
        size = max(stimulus.size(0), perspective.size(0))

        rmat = self.rmat(perspective).expand(size, -1, -1)
        rays = self.monitor.rays(size, height, width)
        grid = self.retina.project(rays, rmat)

        pixels = self.retina_pixel.inverse(stimulus).expand(size, -1, -1, -1)
        pixels = isotropic_grid_sample_2d(pixels, grid=grid, pad_mode=pad_mode)
        pixels = self.monitor_pixel.inverse(pixels)

        return pixels
