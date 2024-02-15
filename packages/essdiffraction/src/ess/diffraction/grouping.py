# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from scippneutron.conversion.graph import beamline

from .types import (
    DspacingBins,
    DspacingData,
    DspacingHistogram,
    FocussedData,
    NormalizedByVanadium,
    RunType,
    TwoThetaBins,
)


def group_by_two_theta(
    data: DspacingData[RunType], edges: TwoThetaBins
) -> FocussedData[RunType]:
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
    return FocussedData[RunType](
        out.bin(two_theta=edges.to(unit=out.coords['two_theta'].unit, copy=False))
    )


def merge_all_pixels(data: DspacingData[RunType]) -> FocussedData[RunType]:
    """Combine all pixels (spectra) of the detector.

    Parameters
    ----------
    data:
        Input data with a `'spectrum'` dimension.

    Returns
    -------
    :
        The input without a `'spectrum'` dimension.
    """
    return FocussedData(data.bins.concat('spectrum'))


def finalize_histogram(
    data: NormalizedByVanadium, edges: DspacingBins
) -> DspacingHistogram:
    """Finalize the d-spacing histogram.

    Histograms the input data into the given d-spacing bins.

    Parameters
    ----------
    data:
        Data to be histogrammed.
    edges:
        Bin edges in d-spacing.

    Returns
    -------
    :
        Histogrammed data.
    """
    return DspacingHistogram(data.hist(dspacing=edges))


providers = (merge_all_pixels, finalize_histogram)
"""Sciline providers for grouping pixels."""
