# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from itertools import pairwise
from typing import NewType

import numpy as np
import scipp as sc
from ess.reduce.nexus.types import FilePath
from h5py import File, Group

from .types import (
    DEFAULT_HISTOGRAM_PATH,
    HistogramModeDetectorsPath,
    ImageDetectorName,
    ImageKeyLogs,
    RotationLogs,
    RotationMotionSensorName,
)

HistogramModeDetectorData = NewType("HistogramModeDetectorData", sc.DataArray)
"""Histogram mode detector data."""
ImageKeyCoord = NewType("ImageKeyCoord", sc.Variable)
"""Image key coordinate."""
ImageStacks = NewType("ImageStacks", sc.DataArray)
"""Image stacks separated by ImageKey values via timestamp."""
RotationAngleCoord = NewType("RotationAngleCoord", sc.Variable)
"""Rotation angle coordinate."""


def _data_with_timestamp(gr: Group, data_unit: str = 'dimensionless') -> sc.DataArray:
    return sc.DataArray(
        data=sc.array(
            dims=["time"],
            values=gr['value'][()].astype(int, copy=False),
            unit=data_unit,
        ),
        coords={
            "time": sc.datetimes(
                dims=["time"],
                values=gr["time"][()].astype(int, copy=False),
                unit='ns',
            )
        },
    )


def load_nexus_histogram_mode_detector_data(
    *,
    file_path: FilePath,
    image_detector_name: ImageDetectorName,
    histogram_mode_detectors_path: HistogramModeDetectorsPath = DEFAULT_HISTOGRAM_PATH,
) -> HistogramModeDetectorData:
    with File(file_path, mode="r") as f:
        detector_dataset = f[
            f"{histogram_mode_detectors_path}/{image_detector_name}/data"
        ]
        x_length, y_length = detector_dataset["value"].shape[1:]
        return HistogramModeDetectorData(
            sc.sort(
                sc.DataArray(
                    data=sc.array(
                        dims=["time", "x", "y"],
                        values=detector_dataset['value'][()].astype(
                            np.int32, copy=False, casting='safe'
                        ),
                        unit='counts',
                    ),
                    coords={
                        "time": sc.datetimes(
                            dims=["time"],
                            values=detector_dataset["time"][()],
                            unit='ns',
                        ),
                        "x": sc.arange(dim="x", start=0, stop=x_length),
                        "y": sc.arange(dim="y", start=0, stop=y_length),
                    },
                ),
                'time',
            )
        )
        # det = detector_dataset['value'][()]
        # return HistogramModeDetectorData(
        #     sc.sort(
        #         sc.DataArray(
        #             data=sc.array(
        #                 dims=["time"],  # , "x", "y"],
        #                 values=list(range(det.shape[0])),
        #                 unit='counts',
        #                 # dtype=object,
        #             ),
        #             coords={
        #                 "time": sc.datetimes(
        #                     dims=["time"],
        #                   values=detector_dataset["time"][()].astype(int, copy=False),
        #                     unit='ns',
        #                 ),
        #                 # "x": sc.arange(dim="x", start=0, stop=x_length),
        #                 # "y": sc.arange(dim="y", start=0, stop=y_length),
        #             },
        #         ),
        #         'time',
        #     )
        # )


def load_image_key_logs(
    *,
    file_path: FilePath,
    image_detector_name: ImageDetectorName,
    histogram_mode_detectors_path: HistogramModeDetectorsPath = DEFAULT_HISTOGRAM_PATH,
) -> ImageKeyLogs:
    with File(file_path, mode="r") as f:
        detector_image_keys = f[
            f"{histogram_mode_detectors_path}/{image_detector_name}/image_key"
        ]
        return ImageKeyLogs(_data_with_timestamp(detector_image_keys))


def load_nexus_rotation_logs(
    file_path: FilePath,
    motion_sensor_name: RotationMotionSensorName,
) -> RotationLogs:
    with File(file_path, mode="r") as f:
        rotations = f[f"entry/instrument/{motion_sensor_name}/rotation_stage_readback"]
        return RotationLogs(
            sc.DataArray(
                data=sc.array(
                    dims=["time"],
                    values=rotations['value'][()],
                    unit='degrees',
                ),
                coords={
                    "time": sc.datetimes(
                        dims=["time"],
                        values=rotations["time"][()].astype(int, copy=False),
                        unit='ns',
                    )
                },
            )
        )


def derive_log_coord_by_range(
    da: sc.DataArray, log: sc.DataArray, out_of_range_value: sc.Variable
) -> sc.Variable:
    """Sort the logs by time and decide which log entry corresponds to each time bin.

    It assumes a log value is valid until the next log entry.
    """
    sorted_logs = sc.sort(log, 'time')
    min_time = sc.datetime(da.coords['time'].min().value - 1, unit='ns')
    max_time = sc.datetime(da.coords['time'].max().value + 1, unit='ns')
    time_edges = sc.concat((min_time, sorted_logs.coords['time'], max_time), 'time')
    sorted_keys = sc.concat((out_of_range_value, sorted_logs.data), 'time')
    return sc.concat(
        [
            sc.broadcast(
                cur_key,
                dims=['time'],
                shape=da['time', start:end].coords['time'].shape,
            )
            for (start, end), cur_key in zip(
                pairwise(time_edges), sorted_keys, strict=True
            )
        ],
        'time',
    )


def derive_image_key_coord(
    histograms: HistogramModeDetectorData, image_keys: ImageKeyLogs
) -> ImageKeyCoord:
    return ImageKeyCoord(
        derive_log_coord_by_range(histograms, image_keys, sc.scalar(-1))
    )


def derive_rotation_angle_coord(
    histograms: HistogramModeDetectorData, rotation_angles: RotationLogs
) -> RotationAngleCoord:
    return RotationAngleCoord(
        derive_log_coord_by_range(
            histograms, rotation_angles, sc.scalar(-1.0, unit=rotation_angles.data.unit)
        )
    )


def apply_logs_as_coords(
    histograms: HistogramModeDetectorData,
    rotation_angles: RotationAngleCoord,
    image_keys: ImageKeyCoord,
) -> ImageStacks:
    copied = histograms.copy(deep=False)
    copied.coords['image_key'] = image_keys
    copied.coords['rotation_angle'] = rotation_angles
    return ImageStacks(copied)
