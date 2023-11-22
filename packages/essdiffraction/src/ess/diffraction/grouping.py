# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from scippneutron.conversion.graph import beamline

from .types import (
    DspacingBins,
    DspacingData,
    DspacingHistogram,
    MergedPixels,
    NormalizedByVanadium,
    RunType,
    TwoThetaBins,
)


def group_by_two_theta(
    data: DspacingData[RunType], edges: TwoThetaBins
) -> MergedPixels[RunType]:
    """
    Group data into two_theta bins.

    Parameters
    ----------
    data:
        Input data array with events. Must contain a coord called 'two_theta'
        or coords that can be used to compute it.
    edges:
        Bin edges in two_theta. `data` is grouped into those bins.

    Returns
    -------
    :
        `data` grouped into two_theta bins.
    """
    out = data.transform_coords('two_theta', graph=beamline.beamline(scatter=True))
    return MergedPixels[RunType](
        out.bin(two_theta=edges.to(unit=out.coords['two_theta'].unit, copy=False))
    )


def merge_all_pixels(data: DspacingData[RunType]) -> MergedPixels[RunType]:
    """Combine all pixels (spectra) of the detector."""
    return MergedPixels(data.bins.concat('spectrum'))


def finalize_histogram(
    data: NormalizedByVanadium, edges: DspacingBins
) -> DspacingHistogram:
    """Finalize the d-spacing histogram."""
    return DspacingHistogram(data.hist(dspacing=edges))


providers = (merge_all_pixels, finalize_histogram)
"""Sciline providers for grouping pixels."""
