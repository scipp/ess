# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import warnings
from typing import NewType

import scipp as sc

from .io import (
    TIME_COORD_NAME,
    DarkCurrentImageStacks,
    OpenBeamImageStacks,
    SampleImageStacksWithLogs,
)

AverageBackgroundPixelCounts = NewType("AverageBackgroundPixelCounts", sc.Variable)
"""mean(background)."""
AverageSamplePixelCounts = NewType("AverageSamplePixelCounts", sc.Variable)
"""mean(sample)."""
ScaleFactor = NewType("ScaleFactor", sc.Variable)
"""AverageBackgroundPixelCounts / AverageSamplePixelCounts."""

OpenBeamImage = NewType("OpenBeamImage", sc.DataArray)
"""Open beam image. mean(OpenBeam)"""
DarkCurrentImage = NewType("DarkCurrentImage", sc.DataArray)
"""Dark current image. mean(DarkCurrentImages)"""
CleansedOpenBeamImage = NewType("CleansedOpenBeamImage", sc.DataArray)
"""OpenBeam - DarkCrrent"""
CleansedSampleImages = NewType("CleansedSampleImages", sc.DataArray)
"""SampleImageStack - DarkCurrent"""
SampleImageStacks = NewType("SampleImageStacks", sc.DataArray)
"""Sample image stack ready to be used for normalization."""
BackgroundImage = NewType("BackgroundImage", sc.DataArray)
"""Background image ready to be used for normalization."""
NormalizedSampleImages = NewType("NormalizedSampleImages", sc.DataArray)
"""Normalized sample image stack. SampleImages / Background * ScaleFactor"""


BackgroundPixelThreshold = NewType("BackgroundPixelThreshold", sc.Variable)
"""Threshold of the background pixel values."""
SamplePixelThreshold = NewType("SamplePixelThreshold", sc.Variable)
"""Threshold of the sample pixel values."""


def _warn_constant_exposure_time(target: str) -> None:
    warning_message = f"Computing {target.strip()} assuming constant exposure time."
    warnings.warn(warning_message, stacklevel=1)


def _mean_all_dims(data: sc.Variable) -> sc.Variable:
    """Calculate the mean of all dimensions one by one to avoid overflow."""
    if data.shape == ():  # scalar
        return data
    return _mean_all_dims(data.mean(dim=data.dims[0]))


def average_open_beam_images(open_beam: OpenBeamImageStacks) -> OpenBeamImage:
    """Average the open beam image stack.

    .. math::

        OpenBeam = mean(OpenBeam, dim=\\text{'time'})

    """
    _warn_constant_exposure_time("average open beam image")
    return OpenBeamImage(sc.mean(open_beam, dim=TIME_COORD_NAME))


def average_dark_current_images(
    dark_current: DarkCurrentImageStacks,
) -> DarkCurrentImage:
    """Average the dark current image stack.

    .. math::

        DarkCurrent = mean(DarkCurrent, dim=\\text{'time'})

    """
    _warn_constant_exposure_time("average dark current image")
    return DarkCurrentImage(sc.mean(dark_current, dim=TIME_COORD_NAME))


def cleanse_open_beam_image(
    open_beam: OpenBeamImage, dark_current: DarkCurrentImage
) -> CleansedOpenBeamImage:
    """Calculate the background image stack.

    .. math::

        Background = OpenBeam - DarkCurrent

    Parameters
    ----------
    open_beam:
        Open beam image.

    dark_current:
        Dark current image.

    """
    return CleansedOpenBeamImage(open_beam - dark_current)


def cleanse_sample_images(
    sample_images: SampleImageStacksWithLogs, dark_current: DarkCurrentImage
) -> CleansedSampleImages:
    """Cleanse the sample image stack.

    We subtract the averaged dark current image from the sample image stack.

    .. math::

        CleansedSample_{i} = Sample_{i} - mean(DarkCurrent, dim=\\text{'time'})

        \\text{where } i \\text{ is an index of an image.}

    Parameters
    ------
    sample_images:
        Sample image stack.

    dark_current:
        Dark current image.

    """
    return CleansedSampleImages(sample_images - dark_current)


def average_background_pixel_counts(
    background: BackgroundImage,
) -> AverageBackgroundPixelCounts:
    """Calculate the average background pixel counts."""
    _warn_constant_exposure_time("average background pixel counts")
    return AverageBackgroundPixelCounts(background.data.mean())


def average_sample_pixel_counts(
    sample_images: SampleImageStacks,
) -> AverageSamplePixelCounts:
    """Calculate the average sample pixel counts.

    Notes
    -----
    For performance reason, we tried calculating
    the mean of sample images and dark current images
    first and subtract them afterwards,
    instead of using the subtracted image stack directly.
    It was to utilize that the integer operation is faster than
    the floating point operation.

    However, we are ceiling negative values to zero
    after cleansing the sample images with dark current images.

    Therefore we need to calculate the mean of the cleansed sample images
    to avoid negative values in the average calculation.

    We don't calculate ``mean(cleansed_sample_images)`` at once
    since it is a large array and it may cause memory issues.

    There was an example of 361 images of 2048x2048 pixels with 32-bit integer data
    exceeded the limit of the maximum integer so the average calculation failed
    and returned negative values.
    """
    _warn_constant_exposure_time("average sample pixel counts")
    return AverageSamplePixelCounts(_mean_all_dims(sample_images.data))


def calculate_scale_factor(
    average_bg: AverageBackgroundPixelCounts, average_sample: AverageSamplePixelCounts
) -> ScaleFactor:
    """Calculate the scale factor from average background and sample pixel counts.

    .. math::

            ScaleFactor = AverageBackgroundPixelCounts / AverageSamplePixelCounts

    """
    return ScaleFactor(average_bg / average_sample)


def apply_threshold_to_sample_images(
    samples: CleansedSampleImages, sample_threshold: SamplePixelThreshold
) -> SampleImageStacks:
    """Apply the threshold to the sample image stack.

    Parameters
    ----------
    samples:
        Sample image stack.

    sample_threshold:
        Threshold for the sample pixel values.
        Any pixel values less than ``sample_threshold``
        are replaced with ``sample_threshold``.

    """
    samples = CleansedSampleImages(samples.copy(deep=False))
    samples.data = sc.where(
        samples.data < sample_threshold, sample_threshold, samples.data
    )
    return SampleImageStacks(samples)


def apply_threshold_to_background_image(
    background: CleansedOpenBeamImage, background_threshold: BackgroundPixelThreshold
) -> BackgroundImage:
    """Apply the threshold to the background image.

    Parameters
    ----------
    background:
        Background image.

    background_threshold:
        Threshold for the background pixel values.
        Any pixel values less than ``background_threshold``
        are replaced with ``background_threshold``.

    """
    background = CleansedOpenBeamImage(background.copy(deep=False))
    background.data = sc.where(
        background.data < background_threshold, background_threshold, background.data
    )
    return BackgroundImage(background)


def normalize_sample_images(
    *, samples: SampleImageStacks, background: BackgroundImage, factor: ScaleFactor
) -> NormalizedSampleImages:
    """Normalize the sample image stack.

    .. math::

            NormalizedImages = SampleImages / Background * ScaleFactor

    Parameters
    ----------

    samples:
        Sample image stack to be normalized.

    background:
        Background image to be used for normalization.

    factor:
        Scale factor for the normalization.


    Raises
    ------
    ValueError:
        If the scale factor is negative.
        It is for the safety of the calculation on short data type.
        Depending on how you calculate the scale factor,
        the operation might fail and return negative values.

    """
    if factor < 0:
        raise ValueError(f"Scale factor must be positive, but got {factor}.")
    _warn_constant_exposure_time("normalized sample image stack")
    # For performance reason, background / factor is calculated first.
    return NormalizedSampleImages(samples / (background / factor))
