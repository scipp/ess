# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import sciline as sl

from .io import (
    MaxDim1,
    MaxDim2,
    MinDim1,
    MinDim2,
    apply_logs_as_coords,
    load_nexus_histogram_mode_detector,
    load_nexus_rotation_logs,
    retrieve_dark_current_images,
    retrieve_open_beam_images,
    retrieve_sample_images,
    separate_detector_images,
    separate_image_by_keys,
    separate_image_key_logs,
)
from .normalize import (
    average_non_sample_images,
    calculate_d,
    calculate_d0,
    calculate_d_factor,
    calculate_white_beam_background,
    cleanse_sample_images,
    grouped_images_by_rotation_angle,
    normalize_sample_images,
)
from .types import (
    DEFAULT_HISTOGRAM_PATH,
    HistogramModeDetectorsPath,
    ImageDetectorName,
    RotationMotionSensorName,
)

_IO_PROVIDERS = (
    apply_logs_as_coords,
    load_nexus_histogram_mode_detector,
    load_nexus_rotation_logs,
    retrieve_dark_current_images,
    retrieve_open_beam_images,
    retrieve_sample_images,
    separate_detector_images,
    separate_image_by_keys,
    separate_image_key_logs,
)
_NORMALIZATION_PROVIDERS = (
    average_non_sample_images,
    calculate_d,
    calculate_d0,
    calculate_d_factor,
    calculate_white_beam_background,
    cleanse_sample_images,
    grouped_images_by_rotation_angle,
    normalize_sample_images,
)

def YmirWorkflow() -> sl.Pipeline:
    return sl.Pipeline(
        (*_IO_PROVIDERS, *_NORMALIZATION_PROVIDERS),
        params={
            MinDim1: MinDim1(None),
            MaxDim1: MaxDim1(None),
            MinDim2: MinDim2(None),
            MaxDim2: MaxDim2(None),
            HistogramModeDetectorsPath: DEFAULT_HISTOGRAM_PATH,
            ImageDetectorName: ImageDetectorName('orca'),
            RotationMotionSensorName: RotationMotionSensorName('motion_cabinet_2'),
        },
    )
