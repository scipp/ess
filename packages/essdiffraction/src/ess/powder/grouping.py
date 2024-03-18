# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Grouping and merging of pixels / voxels."""

from scippneutron.conversion.graph import beamline

from .types import (
    DetectorDimensions,
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


def merge_all_pixels(
    data: DspacingData[RunType], dims: DetectorDimensions
) -> FocussedData[RunType]:
    """Combine all pixels (spectra) of the detector.

    Parameters
    ----------
    data:
        Input data with pixel dimensions.
    dims:
        The dimensions to reduce over.
        Corresponds to the pixel dimensions of the detector.

    Returns
    -------
    :
        The input without pixel dimensions.
    """
    # Put the dims into the same order as in the data.
    # See https://github.com/scipp/scipp/issues/3408
    to_concat = tuple(dim for dim in data.dims if dim in dims)
    return FocussedData(data.bins.concat(to_concat))


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
