# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from typing import NewType

import scipp as sc

from .io import (
    IMAGE_KEY_COORD_NAME,
    ROTATION_ANGLE_COORD_NAME,
    TIME_COORD_NAME,
    ImageKey,
    ImageStacks,
)

D0 = NewType("D0", sc.Variable)
"""~math:`D0 = mean(background)`."""
D = NewType("D", sc.Variable)
"""~math:`D = mean(sample)`."""
DFactor = NewType("DFactor", sc.Variable)
"""~math:`D factor = D0 / D`."""

AveragedImages = NewType("AveragedImages", sc.DataArray)
"""Averaged images by image key."""
OpenBeamImage = NewType("OpenBeamImage", sc.DataArray)
"""Open beam image. ~math:`average(open_beam)`."""
DarkCurrentImage = NewType("DarkCurrentImage", sc.DataArray)
"""Dark current image ~math:`average(dark_current)`."""
AveragedNonSampleImages = NewType("AveragedNonSampleImages", sc.DataArray)
"""Averaged open beam and dark current images."""
BackgroundImage = NewType("BackgroundImage", sc.DataArray)
"""Background image stack. ~math:`background = open_beam - dark_current`."""
CleansedSampleImages = NewType("CleansedSampleImages", sc.DataArray)
"""Sample image stack - dark current."""
NormalizedSampleImages = NewType("NormalizedSampleImages", sc.DataArray)
"""Normalized sample image stack.

~math:`normalized_sample = sample / background * DFactor`.
"""
GroupedSampleImages = NewType("GroupedSampleImages", sc.DataArray)
"""Grouped sample image stack by rotation angle."""


def _select_images_by_key(
    image_stacks: sc.DataArray, key: ImageKey, *keys: ImageKey
) -> sc.DataArray:
    coord = image_stacks.coords[IMAGE_KEY_COORD_NAME]
    indices = [ImageKey.as_index(_key, image_stacks) for _key in (key, *keys)]
    selections = coord == indices.pop(0)
    for idx in indices:
        selections |= coord == idx
    return image_stacks[selections]


def average_non_sample_images(image_stacks: ImageStacks) -> AveragedNonSampleImages:
    bg_images = _select_images_by_key(
        image_stacks, ImageKey.OPEN_BEAM, ImageKey.DARK_CURRENT
    )
    return AveragedNonSampleImages(
        bg_images.groupby(IMAGE_KEY_COORD_NAME).mean(dim=TIME_COORD_NAME)
    )


def _select_image_by_key(image_stacks: sc.DataArray, key: ImageKey) -> sc.DataArray:
    return image_stacks[
        IMAGE_KEY_COORD_NAME,
        ImageKey.as_index(key, image_stacks),
    ]


def calculate_white_beam_background(
    averaged: AveragedNonSampleImages,
) -> BackgroundImage:
    """Calculate the background image stack.

    We average the open beam and dark current image stack
    to create the single background image.

    .. math::

        Background = mean(OpenBeam) - mean(DarkCurrent)

    """
    open_beam = _select_image_by_key(averaged, ImageKey.OPEN_BEAM)
    dark_current = _select_image_by_key(averaged, ImageKey.DARK_CURRENT)
    return BackgroundImage(open_beam - dark_current)


def cleanse_sample_images(
    image_stacks: ImageStacks, bg_images: AveragedNonSampleImages
) -> CleansedSampleImages:
    """Cleanse the sample image stack.

    We subtract the averaged dark current image from the sample image stack.

    .. math::

        CleansedSample_{i} = Sample_{i} - mean(DarkCurrent, dim='time')

        \\text{where } i \\text{ is an index of an image.}

    """
    sample_images = _select_images_by_key(image_stacks, ImageKey.SAMPLE)
    dark_current = _select_image_by_key(bg_images, ImageKey.DARK_CURRENT)
    return CleansedSampleImages(sample_images - dark_current)


def calculate_d0(background: BackgroundImage) -> D0:
    """Calculate the D0 value from background image stack.

    :math:`D0 = mean(background counts of all pixels)`
    """
    return D0(sc.mean(background))


def calculate_d(samples: CleansedSampleImages) -> D:
    """Calculate the D value from the sample image stack.

    :math:`D = mean(sample counts of all pixels)`
    """
    return D(samples.data.mean())


def calculate_d_factor(d0: D0, d: D) -> DFactor:
    """Calculate the D factor from D0 and D.

    :math:`DFactor = D0 / D`
    """
    return DFactor(d0 / d)


def normalize_sample_images(
    samples: CleansedSampleImages, d_factor: DFactor, background: BackgroundImage
) -> NormalizedSampleImages:
    """Normalize the sample image stack.

    .. math::

        NormalizedSample_{i} = CleansedSample_{i} / Background * DFactor


    .. math::

        DFactor = D0 / D

        D0 = mean(Background)  \\text{  *of all pixels}

        D = mean(Sample)  \\text{  *of all pixels}


    .. math::

        CleansedSample_{i} = Sample_{i} - mean(DarkCurrent, dim=\\text{'time'})

        \\text{where } i \\text{ is an index of an image.}

    """
    return NormalizedSampleImages((samples / background) * d_factor)


def grouped_images_by_rotation_angle(
    normalized_samples: NormalizedSampleImages,
) -> GroupedSampleImages:
    """Group the normalized sample images by rotation angle."""
    return GroupedSampleImages(
        normalized_samples.groupby(ROTATION_ANGLE_COORD_NAME).mean(dim=TIME_COORD_NAME)
    )
