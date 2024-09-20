# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import sciline as sl

from .io import (
    DEFAULT_FILE_LOCK,
    FileLock,
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
    DEFAULT_BACKGROUND_THRESHOLD,
    DEFAULT_SAMPLE_THRESHOLD,
    BackgroundPixelThreshold,
    SamplePixelThreshold,
    apply_threshold_to_background_image,
    apply_threshold_to_sample_images,
    average_background_pixel_counts,
    average_dark_current_images,
    average_open_beam_images,
    average_sample_pixel_counts,
    calculate_scale_factor,
    cleanse_open_beam_image,
    cleanse_sample_images,
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
    apply_threshold_to_background_image,
    apply_threshold_to_sample_images,
    average_background_pixel_counts,
    average_dark_current_images,
    average_open_beam_images,
    average_sample_pixel_counts,
    calculate_scale_factor,
    cleanse_open_beam_image,
    cleanse_sample_images,
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
            BackgroundPixelThreshold: DEFAULT_BACKGROUND_THRESHOLD,
            SamplePixelThreshold: DEFAULT_SAMPLE_THRESHOLD,
            FileLock: DEFAULT_FILE_LOCK,
        },
    )
