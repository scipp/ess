# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

from ess.imaging.data import get_ymir_images_path
from ess.imaging.io import (
    FilePath,
    ImageDetectorName,
    RotationMotionSensorName,
    load_nexus_histogram_mode_detector,
    load_nexus_rotation_logs,
)
from ess.imaging.types import DEFAULT_HISTOGRAM_PATH


def test_nexus_histogram_mode_detector_loading_warnings() -> None:
    with pytest.warns(expected_warning=UserWarning, match='The unit of the histogram'):
        # We may need to remove the manual assignment of the unit
        # once the files are updated to have the correct unit
        assert isinstance(
            load_nexus_histogram_mode_detector(
                file_path=FilePath(get_ymir_images_path()),
                image_detector_name=ImageDetectorName('orca'),
                histogram_mode_detectors_path=DEFAULT_HISTOGRAM_PATH,
            ),
            sc.DataGroup,
        )


def test_nexus_rotation_logs_loading() -> None:
    assert isinstance(
        load_nexus_rotation_logs(
            file_path=FilePath(get_ymir_images_path()),
            motion_sensor_name=RotationMotionSensorName('motion_cabinet_2'),
        ),
        sc.DataArray,
    )
