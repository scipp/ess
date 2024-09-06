# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import warnings
from collections.abc import Callable, Iterable
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import NewType

import scipp as sc
import scippnexus as snx
from tifffile import imwrite

from ess.reduce.nexus.types import FilePath

from .types import (
    DEFAULT_HISTOGRAM_PATH,
    HistogramModeDetectorsPath,
    ImageDetectorName,
    ImageKeyLogs,
    RotationLogs,
    RotationMotionSensorName,
)

HistogramModeDetector = NewType("HistogramModeDetector", sc.DataGroup)
"""Histogram mode detector data group."""
HistogramModeDetectorData = NewType("HistogramModeDetectorData", sc.DataArray)
"""Histogram mode detector data."""
ImageKeyCoord = NewType("ImageKeyCoord", sc.Variable)
"""Image key coordinate."""
ImageStacks = NewType("ImageStacks", sc.DataArray)
"""Image stacks separated by ImageKey values via timestamp."""
RotationAngleCoord = NewType("RotationAngleCoord", sc.Variable)
"""Rotation angle coordinate."""


class ImageKey(Enum):
    """Image key values."""

    SAMPLE = 0
    OPEN_BEAM = 1
    DARK_CURRENT = 2


def load_nexus_histogram_mode_detector(
    *,
    file_path: FilePath,
    image_detector_name: ImageDetectorName,
    histogram_mode_detectors_path: HistogramModeDetectorsPath = DEFAULT_HISTOGRAM_PATH,
) -> HistogramModeDetector:
    with snx.File(file_path, mode="r") as f:
        img_path = f"{histogram_mode_detectors_path}/{image_detector_name}"
        dg: sc.DataGroup = f[img_path][()]

    # Manually assign unit to the histogram detector mode data
    img: sc.DataArray = dg['data']
    if (original_unit := img.unit) != 'counts':
        img.unit = 'counts'
        warnings.warn(
            f"The unit of the histogram detector data is [{original_unit}]. "
            f"It is expected to be [{img.unit}]. "
            f"The loader manually assigned the unit to be [{img.unit}].",
            stacklevel=0,
        )
    return HistogramModeDetector(dg)


def separate_detector_images(dg: HistogramModeDetector) -> HistogramModeDetectorData:
    return HistogramModeDetectorData(sc.sort(dg['data'], 'time'))


def separate_image_key_logs(*, dg: HistogramModeDetector) -> ImageKeyLogs:
    return ImageKeyLogs(dg['image_key']['value'])


def load_nexus_rotation_logs(
    file_path: FilePath,
    motion_sensor_name: RotationMotionSensorName,
) -> RotationLogs:
    log_path = f"entry/instrument/{motion_sensor_name}/rotation_stage_readback"
    with snx.File(file_path, mode="r") as f:
        return RotationLogs(f[log_path][()]['value'])


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
    out_of_range = sc.scalar(-1, dtype=image_keys.data.dtype, unit=image_keys.data.unit)
    return ImageKeyCoord(
        derive_log_coord_by_range(histograms, image_keys, out_of_range)
    )


def derive_rotation_angle_coord(
    histograms: HistogramModeDetectorData, rotation_angles: RotationLogs
) -> RotationAngleCoord:
    return RotationAngleCoord(
        derive_log_coord_by_range(
            histograms,
            rotation_angles,
            sc.scalar(
                -1.0, unit=rotation_angles.data.unit, dtype=rotation_angles.data.dtype
            ),
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


DEFAULT_IMAGE_NAME_PREFIX_MAP = {
    ImageKey.SAMPLE.value: "sample",
    ImageKey.DARK_CURRENT.value: "dc",
    ImageKey.OPEN_BEAM.value: "ob",
}


def dummy_progress_wrapper(core_iterator: Iterable) -> Iterable:
    yield from core_iterator


def _save_merged_images(
    *, image_stacks: ImageStacks, image_prefix: str, output_dir: Path
) -> None:
    image_path = output_dir / Path(
        f"{image_prefix}_0000_{image_stacks.sizes['time']:04d}.tiff"
    )
    imwrite(image_path, image_stacks.values)


def _save_individual_images(
    *,
    image_stacks: ImageStacks,
    image_prefix: str,
    output_dir: Path,
    progress_wrapper: Callable[[Iterable], Iterable] = dummy_progress_wrapper,
) -> None:
    for i_image in progress_wrapper(range(image_stacks.sizes['time'])):
        cur_image = image_stacks['time', i_image]
        image_path = output_dir / Path(f"{image_prefix}_{i_image:04d}.tiff")
        imwrite(image_path, cur_image.values)


def _validate_output_dir(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        # make sure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=False)
    elif not output_dir.is_dir():
        raise ValueError(f"Output directory {output_dir} is not a directory.")
    elif len(list(output_dir.iterdir())) > 0:
        raise RuntimeError(f"Output directory {output_dir} is not empty.")


def export_image_stacks_as_tiff(
    *,
    output_dir: str | Path,
    image_stacks: ImageStacks,
    merge_image_by_key: bool,
    overwrite: bool,
    progress_wrapper: Callable[[Iterable], Iterable] = dummy_progress_wrapper,
    image_prefix_map: dict[int, str] = DEFAULT_IMAGE_NAME_PREFIX_MAP,
) -> None:
    """Save images into disk.

    Parameters
    ----------
    output_dir:
        Output directory to save images.

    image_stacks:
        Image stacks to save.

    merge_image_by_key:
        Flag to merge images into one file.

    overwrite:
        Flag to overwrite existing files.
        If True, it will clear the output directory before saving images.

    image_prefix_map:
        Map of image name prefixes to their corresponding image key.

    """
    # Remove existing files if overwrite is True
    if (
        overwrite
        and (output_path := Path(output_dir)).exists()
        and output_path.is_dir()
    ):
        for file in output_path.iterdir():
            file.unlink()

    _validate_output_dir(output_path)

    image_keys = image_stacks.coords['image_key']
    for image_key in progress_wrapper(set(image_keys.values)):
        cur_key = sc.scalar(image_key, unit=image_keys.unit, dtype=image_keys.dtype)
        cur_images = image_stacks[image_stacks.coords['image_key'] == cur_key]
        if merge_image_by_key:
            _save_merged_images(
                image_stacks=ImageStacks(cur_images),
                image_prefix=image_prefix_map[image_key],
                output_dir=output_path,
            )
        else:
            _save_individual_images(
                image_stacks=ImageStacks(cur_images),
                image_prefix=image_prefix_map[image_key],
                output_dir=output_path,
                progress_wrapper=progress_wrapper,
            )
