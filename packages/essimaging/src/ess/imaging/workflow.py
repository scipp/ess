# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import sciline as sl
import scipp as sc

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
_DEFAULT_BACKGROUND_THRESHOLD = BackgroundPixelThreshold(sc.scalar(1.0, unit="counts"))
_DEFAULT_SAMPLE_THRESHOLD = SamplePixelThreshold(sc.scalar(0.0, unit="counts"))


def YmirImageNormalizationWorkflow() -> sl.Pipeline:
    """
    Ymir histogram mode imaging normalization workflow.

    Default Normalization Formula
    -----------------------------

    .. math::

        NormalizedSample_{i} = SampleImageStacks_{i} / BackgroundImage * ScaleFactor


    .. math::

        ScaleFactor = AverageBackgroundPixelCounts / AverageSamplePixelCounts


    .. math::

        SampleImageStacks_{i} = Sample_{i} - mean(DarkCurrent, dim=\\text{'time'})

        \\text{where } i \\text{ is an index of an image.}

        \\text{Pixel values less than sample_threshold}

        \\text{ are replaced with sample_threshold}.

    .. math::

        BackgroundImage = mean(OpenBeam, dim=\\text{'time'})
        - mean(DarkCurrent, dim=\\text{'time'})

        \\text{Pixel values less than } \\text{background_threshold}

        \\text{ are replaced with } \\text{background_threshold}.
    """
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
            BackgroundPixelThreshold: _DEFAULT_BACKGROUND_THRESHOLD,
            SamplePixelThreshold: _DEFAULT_SAMPLE_THRESHOLD,
            FileLock: DEFAULT_FILE_LOCK,
        },
    )
