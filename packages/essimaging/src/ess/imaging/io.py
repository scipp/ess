# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import warnings
from collections.abc import Callable, Generator, Iterable
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

FileLock = NewType("FileLock", bool)
"""File lock mode for reading nexus file."""
DEFAULT_FILE_LOCK = FileLock(True)

HistogramModeDetector = NewType("HistogramModeDetector", sc.DataGroup)
"""Histogram mode detector data group."""
HistogramModeDetectorData = NewType("HistogramModeDetectorData", sc.DataArray)
"""Histogram mode detector data."""
ImageKeyCoord = NewType("ImageKeyCoord", sc.Variable)
"""Image key coordinate."""
SampleImageStacksWithLogs = NewType("SampleImageStacksWithLogs", sc.DataArray)
"""Raw image stacks separated by ImageKey values via timestamp."""
RotationAngleCoord = NewType("RotationAngleCoord", sc.Variable)
"""Rotation angle coordinate."""

RawSampleImageStacks = NewType("RawSampleImageStacks", sc.DataArray)
"""Sample image stacks."""
OpenBeamImageStacks = NewType("OpenBeamImageStacks", sc.DataArray)
"""Open beam image stacks."""
DarkCurrentImageStacks = NewType("DarkCurrentImageStacks", sc.DataArray)
"""Dark current image stacks."""


IMAGE_KEY_COORD_NAME = "image_key"
"""Image key coordinate name."""
TIME_COORD_NAME = "time"
"""Time coordinate name."""
ROTATION_ANGLE_COORD_NAME = "rotation_angle"
"""Rotation angle coordinate name."""
DIM1_COORD_NAME = "dim_1"
"""Dimension 1 coordinate name."""
DIM2_COORD_NAME = "dim_2"
"""Dimension 2 coordinate name."""


class ImageKey(Enum):
    """Image key values."""

    SAMPLE = 0
    OPEN_BEAM = 1
    DARK_CURRENT = 2

    @classmethod
    def as_index(cls, key: "ImageKey", target_da: sc.DataArray | None) -> sc.Variable:
        if target_da is None:
            return sc.scalar(cls(key).value, unit=None)
        elif IMAGE_KEY_COORD_NAME in target_da.coords:
            return sc.scalar(
                cls(key).value,
                unit=target_da.coords[IMAGE_KEY_COORD_NAME].unit,
                dtype=target_da.coords[IMAGE_KEY_COORD_NAME].dtype,
            )
        else:
            return sc.scalar(cls(key).value, unit=target_da.unit, dtype=target_da.dtype)


def load_nexus_histogram_mode_detector(
    *,
    file_path: FilePath,
    image_detector_name: ImageDetectorName,
    histogram_mode_detectors_path: HistogramModeDetectorsPath = DEFAULT_HISTOGRAM_PATH,
    locking: FileLock = DEFAULT_FILE_LOCK,
) -> HistogramModeDetector:
    try:
        with snx.File(file_path, mode="r", locking=locking) as f:
            img_path = f"{histogram_mode_detectors_path}/{image_detector_name}"
            dg: sc.DataGroup = f[img_path][()]
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied to read the nexus file [{file_path}]. "
            "Please check the permission of the file or the directory. "
            "Consider using the `file_lock` parameter to avoid file locking "
            "if the file system is mounted on a network file system. "
            "and it is safe to read the file without locking."
        ) from e

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


MinDim1 = NewType("MinDim1", sc.Variable | None)
"""Minimum value of the first dimension."""
MaxDim1 = NewType("MaxDim1", sc.Variable | None)
"""Maximum value of the first dimension."""
MinDim2 = NewType("MinDim2", sc.Variable | None)
"""Minimum value of the second dimension."""
MaxDim2 = NewType("MaxDim2", sc.Variable | None)


def _make_coord_if_needed(da: sc.DataArray, dim: str) -> None:
    if dim not in da.coords.keys():
        da.coords[dim] = sc.arange(dim=dim, start=0, stop=da.sizes[dim])


def separate_detector_images(
    dg: HistogramModeDetector,
    min_dim_1: MinDim1,
    max_dim_1: MaxDim1,
    min_dim_2: MinDim2,
    max_dim_2: MaxDim2,
) -> HistogramModeDetectorData:
    da: sc.DataArray = sc.sort(dg['data'], 'time')
    # Assign position coordinates to the detector data
    _make_coord_if_needed(da, DIM1_COORD_NAME)
    _make_coord_if_needed(da, DIM2_COORD_NAME)
    # Crop the detector data by the given coordinates
    da = da[DIM1_COORD_NAME, min_dim_1:max_dim_1][DIM2_COORD_NAME, min_dim_2:max_dim_2]
    return HistogramModeDetectorData(da)


def separate_image_key_logs(*, dg: HistogramModeDetector) -> ImageKeyLogs:
    return ImageKeyLogs(sc.sort(dg['image_key']['value'], key='time'))


def load_nexus_rotation_logs(
    file_path: FilePath,
    motion_sensor_name: RotationMotionSensorName,
    locking: FileLock = DEFAULT_FILE_LOCK,
) -> RotationLogs:
    log_path = f"entry/instrument/{motion_sensor_name}/rotation_stage_readback"
    with snx.File(file_path, mode="r", locking=locking) as f:
        return RotationLogs(f[log_path][()]['value'])


def derive_log_coord_by_range(da: sc.DataArray, log: sc.DataArray) -> sc.Variable:
    """Sort the logs by time and decide which log entry corresponds to each time bin.

    It assumes a log value is valid until the next log entry.
    """
    log = sc.sort(log, TIME_COORD_NAME)
    indices = [*log.coords[TIME_COORD_NAME], None]
    return sc.concat(
        [
            sc.broadcast(
                log.data[TIME_COORD_NAME, i_time],
                dims=[TIME_COORD_NAME],
                shape=(da[TIME_COORD_NAME, start:end].sizes[TIME_COORD_NAME],),
            )
            for i_time, (start, end) in enumerate(pairwise(indices))
        ],
        TIME_COORD_NAME,
    )


def _slice_da_by_keys(
    da: sc.DataArray, image_keys: ImageKeyLogs, image_key: ImageKey
) -> Generator[sc.DataArray, None, None]:
    matching_value = image_key.as_index(image_key, image_keys)
    time_coord = image_keys.coords[TIME_COORD_NAME]
    time_intervals = image_keys.sizes[TIME_COORD_NAME]
    for i_time, (cur_time, image_key) in enumerate(
        zip(time_coord, image_keys.data, strict=True)
    ):
        if image_key == matching_value:
            if i_time == time_intervals - 1:
                yield da[TIME_COORD_NAME, cur_time:]
            else:
                next_time = time_coord[i_time + 1]
                yield da[TIME_COORD_NAME, cur_time:next_time]


def _retrieve_image_stacks_by_key(
    da: HistogramModeDetectorData, image_keys: ImageKeyLogs, image_key: ImageKey
) -> sc.DataArray:
    images = [
        sliced_da
        for sliced_da in _slice_da_by_keys(da, image_keys, image_key)
        if da.sizes[TIME_COORD_NAME] > 0
    ]
    if len(images) == 0:
        raise ValueError(f"No images found for {image_key}.")
    elif len(images) == 1:
        return images[0]
    return sc.concat(images, 'time')


AllImageStacks = NewType("AllImageStacks", dict[ImageKey, sc.DataArray])


def separate_image_by_keys(
    da: HistogramModeDetectorData,
    image_keys: ImageKeyLogs,
) -> AllImageStacks:
    return AllImageStacks(
        {key: _retrieve_image_stacks_by_key(da, image_keys, key) for key in ImageKey}
    )


def retrieve_open_beam_images(
    da: HistogramModeDetectorData, image_keys: ImageKeyLogs
) -> OpenBeamImageStacks:
    return OpenBeamImageStacks(
        _retrieve_image_stacks_by_key(da, image_keys, ImageKey.OPEN_BEAM)
    )


def retrieve_dark_current_images(
    da: HistogramModeDetectorData, image_keys: ImageKeyLogs
) -> DarkCurrentImageStacks:
    return DarkCurrentImageStacks(
        _retrieve_image_stacks_by_key(da, image_keys, ImageKey.DARK_CURRENT)
    )


def retrieve_sample_images(
    da: HistogramModeDetectorData, image_keys: ImageKeyLogs
) -> RawSampleImageStacks:
    return RawSampleImageStacks(
        _retrieve_image_stacks_by_key(da, image_keys, ImageKey.SAMPLE)
    )


def apply_logs_as_coords(
    samples: RawSampleImageStacks, rotation_angles: RotationLogs
) -> SampleImageStacksWithLogs:
    # Make sure the data has the same range as the rotation angle coordinate
    min_log_time = rotation_angles.coords[TIME_COORD_NAME].min(TIME_COORD_NAME)
    sliced = samples[TIME_COORD_NAME, min_log_time:].copy(deep=False)
    if sliced.sizes != samples.sizes:
        warnings.warn(
            "The sample data has been sliced to match the rotation angle coordinate.",
            stacklevel=0,
        )
    rotation_angle_coord = derive_log_coord_by_range(samples, rotation_angles)
    sliced.coords['rotation_angle'] = rotation_angle_coord
    return SampleImageStacksWithLogs(sliced)


DEFAULT_IMAGE_NAME_PREFIX_MAP = {
    ImageKey.SAMPLE: "sample",
    ImageKey.DARK_CURRENT: "dc",
    ImageKey.OPEN_BEAM: "ob",
}


def dummy_progress_wrapper(core_iterator: Iterable) -> Iterable:
    yield from core_iterator


def _save_merged_images(
    *, image_stacks: SampleImageStacksWithLogs, image_prefix: str, output_dir: Path
) -> None:
    image_path = output_dir / Path(
        f"{image_prefix}_0000_{image_stacks.sizes['time']:04d}.tiff"
    )
    imwrite(image_path, image_stacks.values)


def _save_individual_images(
    *,
    image_stacks: SampleImageStacksWithLogs,
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
    elif next(output_dir.iterdir(), None) is not None:
        raise RuntimeError(f"Output directory {output_dir} is not empty.")


def export_image_stacks_as_tiff(
    *,
    output_dir: str | Path,
    image_stacks: AllImageStacks,
    merge_image_by_key: bool,
    overwrite: bool,
    progress_wrapper: Callable[[Iterable], Iterable] = dummy_progress_wrapper,
    image_prefix_map: dict[ImageKey, str] = DEFAULT_IMAGE_NAME_PREFIX_MAP,
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

    for image_key, cur_images in progress_wrapper(image_stacks.items()):
        if merge_image_by_key:
            _save_merged_images(
                image_stacks=SampleImageStacksWithLogs(cur_images),
                image_prefix=image_prefix_map[image_key],
                output_dir=output_path,
            )
        else:
            _save_individual_images(
                image_stacks=SampleImageStacksWithLogs(cur_images),
                image_prefix=image_prefix_map[image_key],
                output_dir=output_path,
                progress_wrapper=progress_wrapper,
            )
