# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)


from ..sans2d.general import (
    get_detector_data,
    get_monitor,
    lab_frame_transform,
    sans2d_tube_detector_pixel_shape,
)
from . import io, masking
from .components import (
    DetectorBankOffset,
    SampleOffset,
    apply_component_user_offsets_to_raw_data,
)
from .io import CalibrationFilename, DataFolder, Filename, PixelMaskFilename
from .masking import PixelMask
from .visualization import plot_flat_detector_xy

providers = (
    (
        get_detector_data,
        get_monitor,
        lab_frame_transform,
        sans2d_tube_detector_pixel_shape,
        apply_component_user_offsets_to_raw_data,
    )
    + io.providers
    + masking.providers
)

del get_detector_data
del get_monitor
del lab_frame_transform
del sans2d_tube_detector_pixel_shape

__all__ = [
    'CalibrationFilename',
    'DataFolder',
    'DetectorBankOffset',
    'Filename',
    'apply_component_user_offsets_to_raw_data',
    'io',
    'masking',
    'PixelMask',
    'PixelMaskFilename',
    'providers',
    'SampleOffset',
    'plot_flat_detector_xy',
]
