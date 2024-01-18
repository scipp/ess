# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)


from ..sans2d.general import (
    get_detector_data,
    get_monitor,
    lab_frame_transform,
    sans2d_tube_detector_pixel_shape,
)
from . import io, mantidio, masking
from .common import transmission_from_background_run, transmission_from_sample_run
from .io import PixelMaskFilename
from .mantidio import CalibrationFilename, Filename
from .masking import PixelMask

providers = (
    (
        get_detector_data,
        get_monitor,
        lab_frame_transform,
        sans2d_tube_detector_pixel_shape,
    )
    + mantidio.providers
    + masking.providers
    + io.providers
)

__all__ = [
    'Filename',
    'PixelMask',
    'PixelMaskFilename',
    'CalibrationFilename',
    'transmission_from_background_run',
    'transmission_from_sample_run',
    'providers',
]
