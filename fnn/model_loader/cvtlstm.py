from fnn.model import (
    networks,
    cores,
    readouts,
    modulations,
    perspectives,
    feedforwards,
    recurrents,
    monitors,
    retinas,
    positions,
    bounds,
    features,
    reductions,
    units,
    pixels,
)
from foundation.utils.video import Video
from foundation.utils.resize import PilResize
from PIL import Image
import numpy as np


def preprocess(frames, times):
    """
    Returns a function that preprocesses a video array
    """
    resize = PilResize(Image.Resampling.BILINEAR)
    assert frames.ndim == 3, "frames must be of shape (frame, height, width)"
    assert frames.dtype == np.uint8, "frames must have dtype uint8"
    video = Video.fromarray(frames, times=times)
    video = resize(video, height=144, width=256)
    stimuli = video.generate(period=1 / 30)
    return stimuli

def model(state_dict, device="cuda", freeze=True):
    n_units = state_dict['readout.position.mu'].shape[0]
    # hyperparameters
    ## core
    feedforward_dense_params = {
        "in_spatial": 6,
        "in_stride": 2,
        "out_channels": 128,
        "block_channels": (32, 64, 128),
        "block_groups": (1, 2, 4),
        "block_layers": (2, 2, 2),
        "block_temporals": (3, 3, 3),
        "block_spatials": (3, 3, 3),
        "block_pools": (2, 2, 1),
        "nonlinear": "gelu",
        "dropout": 0.100000,
    }
    recurrent_cvtlstm_params = {
        "in_channels": 256,
        "out_channels": 128,
        "hidden_channels": 256,
        "common_channels": 512,
        "groups": 8,
        "spatial": 3,
        "init_input": -1.0000,
        "init_forget": 1.0000,
        "dropout": 0.100000,
    }
    ## perspective
    monitor_plane_params = {
        "init_center_x": 0.0000,
        "init_center_y": 0.0000,
        "init_center_z": 0.5000,
        "init_center_std": 0.0500,
        "init_angle_x": 0.0000,
        "init_angle_y": 0.0000,
        "init_angle_z": 0.0000,
        "init_angle_std": 0.0500,
    }
    retina_angular_params = {"degrees": 75.000}
    monitor_pixel_params = {
        "power": 1.7000,
        "scale": 1.0000,
        "offset": 0.0000,
    }
    retina_pixel_params = {
        "max_power": 1.000,
        "init_scale": 4.0000,
        "init_offset": -1.4000,
    }
    perspective_monitorretina_params = {
        "height": 128,
        "width": 192,
        "features": [16, 16],
        "nonlinear": "gelu",
        "dropout": 0.100000,
    }
    ## modulation
    modulation_flatlstm_params = {
        "in_features": 16,
        "out_features": 4,
        "hidden_features": 16,
        "init_input": -1.0000,
        "init_forget": 1.0000,
        "dropout": 0.100000,
    }
    ## readout
    position_gaussian_params = {"init_std": 0.4000}
    feature_norm_params = {"groups": 1}
    ## initialization
    initialization_params = {
        "streams": 4,
        "stimuli": 1,
        "perspectives": 2,
        "modulations": 2,
    }
    # assemble the core
    feedforward = feedforwards.Dense(**feedforward_dense_params)
    recurrent = recurrents.CvtLstm(**recurrent_cvtlstm_params)
    core = cores.FeedforwardRecurrent(
        feedforward=feedforward,
        recurrent=recurrent,
    )
    # assemble the perspective module
    monitor = monitors.Plane(**monitor_plane_params)
    monitor_pixel = pixels.StaticPower(**monitor_pixel_params)
    retina = retinas.Angular(**retina_angular_params)
    retina_pixel = pixels.SigmoidPower(**retina_pixel_params)
    perspective = perspectives.MonitorRetina(
        monitor=monitor,
        monitor_pixel=monitor_pixel,
        retina=retina,
        retina_pixel=retina_pixel,
        **perspective_monitorretina_params,
    )
    # assemble the modulation module
    modulation = modulations.FlatLstm(**modulation_flatlstm_params)
    # assemble the readout module
    position = positions.Gaussian(**position_gaussian_params)
    bound = bounds.Tanh()
    feature = features.Norm(**feature_norm_params)
    readout = readouts.PositionFeature(position=position, bound=bound, feature=feature)
    # assemble the reduce module
    reduce = reductions.Mean()
    # assemble the unit module
    unit = units.Poisson()
    # assemble the network
    network = networks.Visual(
        core=core,
        perspective=perspective,
        modulation=modulation,
        readout=readout,
        reduce=reduce,
        unit=unit,
    )
    # initialize the network
    network._init(**initialization_params, units=n_units)
    network =  network.to(device=device).freeze(freeze)
    network.load_state_dict(state_dict)
    return network
