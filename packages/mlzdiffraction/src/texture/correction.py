# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Correction algorithms for POWTEX texture reduction."""

from typing import cast

from ..powtex.types import EmptyCanRun, EmptyVanadiumCanRun, SampleRun, VanadiumRun

from .types import EmptyCanSubtracted3D, NormalisedTexture3D, Texture3D


def subtract_empty_can_from_sample(
    data: Texture3D[SampleRun],
    background: Texture3D[EmptyCanRun],
) -> EmptyCanSubtracted3D[SampleRun]:
    """Subtract empty-can texture data from the sample."""
    if data is None:
        return cast(EmptyCanSubtracted3D[SampleRun], None)
    if background is None:
        return EmptyCanSubtracted3D[SampleRun](data)
    # What if negative values after subtr?
    return EmptyCanSubtracted3D[SampleRun](data - background)


def subtract_empty_can_from_vanadium(
    data: Texture3D[VanadiumRun],
    background: Texture3D[EmptyVanadiumCanRun],
) -> EmptyCanSubtracted3D[VanadiumRun]:
    """Subtract empty-can texture data from the vanadium run."""
    if data is None:
        return cast(EmptyCanSubtracted3D[VanadiumRun], None)
    if background is None:
        return EmptyCanSubtracted3D[VanadiumRun](data)
    return EmptyCanSubtracted3D[VanadiumRun](data - background)


def normalize_by_vanadium(
    data: EmptyCanSubtracted3D[SampleRun],
    vanadium: EmptyCanSubtracted3D[VanadiumRun],
) -> NormalisedTexture3D:
    """Normalize sample texture data by vanadium texture data."""
    if data is None:
        return cast(NormalisedTexture3D, None)
    if vanadium is None:
        return NormalisedTexture3D(data)
    return NormalisedTexture3D(data / vanadium)


providers = (
    subtract_empty_can_from_sample,
    subtract_empty_can_from_vanadium,
    normalize_by_vanadium,
)
