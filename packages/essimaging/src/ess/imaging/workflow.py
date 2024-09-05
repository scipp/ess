# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import sciline as sl

from .io import (
    apply_logs_as_coords,
    derive_image_key_coord,
    derive_rotation_angle_coord,
    load_nexus_histogram_mode_detector,
    load_nexus_rotation_logs,
    separate_detector_images,
    separate_image_key_logs,
)
from .types import (
    DEFAULT_HISTOGRAM_PATH,
    HistogramModeDetectorsPath,
    ImageDetectorName,
    RotationMotionSensorName,
)


def YmirWorkflow() -> sl.Pipeline:
    return sl.Pipeline(
        (
            apply_logs_as_coords,
            derive_image_key_coord,
            derive_rotation_angle_coord,
            load_nexus_rotation_logs,
            load_nexus_histogram_mode_detector,
            separate_detector_images,
            separate_image_key_logs,
        ),
        params={
            HistogramModeDetectorsPath: DEFAULT_HISTOGRAM_PATH,
            ImageDetectorName: ImageDetectorName('orca'),
            RotationMotionSensorName: RotationMotionSensorName('motion_cabinet_2'),
        },
    )
