import torch
from .modules import Module
from .parameters import Parameter
from .utils import Gaussian3d


# -------------- Visual Stimulus Base --------------


class VisualStimulus(Module):
    """Visual Stimulus"""

    def _init(self, channels, frames, height, width):
        """
        Parameters
        ----------
        channels : int
            number of channels (C)
        frames : int
            number of frames (F)
        height : int
            height in pixels (H)
        width : int
            width in pixels (W)
        """
        self.channels = int(channels)
        self.frames = int(frames)
        self.height = int(height)
        self.width = int(width)

    def forward(self):
        """
        Yields
        ------
        3D Tensor
            [C, H, W] -- stimulus frame
        """
        raise NotImplementedError()

    def penalty(self):
        """
        Returns
        -------
        Tensor
            scalar -- penalty value
        """
        raise NotImplementedError()

    @property
    def video(self):
        """
        Returns
        -------
        4D array
            [F, H, W, C], dtype=np.uint8
        """
        raise NotImplementedError()


# -------------- Visual Stimulus Types --------------


class VisualNlm(VisualStimulus):
    """Visual Stimulus with Non-local means Regularization"""

    def __init__(self, bound, init_value=0, spatial_std=1, temporal_std=1, cutoff=4):
        """
        Parameters
        ----------
        bound : fnn.model.bounds.Bound
            stimulus bound
        init_value : float
            initial pixel value
        spatial_sigma : float
            nlm spatial standard deviation
        temporal_sigma : float
            nlm temporal standard deviation
        cutoff : float
            nlm standard deviation cutoff
        """
        super().__init__()

        assert bound.vmin == 0
        assert bound.vmax == 1
        self.bound = bound
        self.init_value = float(init_value)
        self.gaussian = Gaussian3d(spatial_std=spatial_std, temporal_std=temporal_std, cutoff=cutoff)

    def _init(self, channels, frames, height, width):
        """
        Parameters
        ----------
        channels : int
            number of channels (C)
        frames : int
            number of frames (F)
        height : int
            height in pixels (H)
        width : int
            width in pixels (W)
        """
        super()._init(channels=channels, frames=frames, height=height, width=width)

        self.frames = Parameter(
            torch.full([self.channels, self.frames, self.height, self.width], self.init_value),
        )

    def forward(self):
        """
        Yields
        ------
        3D Tensor
            [C, H, W] -- stimulus frame
        """
        yield from self.bound(self.frames).unbind(dim=1)

    def penalty(self):
        """https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/Buades-NonLocal.pdf

        Returns
        -------
        Tensor
            scalar -- penalty value
        """
        delt = self.frames - self.gaussian(self.frames)
        return delt.pow(2).sum()

    @property
    def video(self):
        """
        Returns
        -------
        4D array
            [F, H, W, C], dtype=np.uint8
        """
        with torch.no_grad():

            video = self.bound(self.frames)
            video = torch.einsum("C F H W -> F H W C", video)
            video = video.mul(255).round().to(device="cpu", dtype=torch.uint8)

            return video.numpy()
