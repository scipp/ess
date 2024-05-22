# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Grouping and merging of pixels / voxels."""

from typing import Optional

from .types import (
    DspacingBins,
    DspacingHistogram,
    FocussedData,
    MaskedData,
    NormalizedByVanadium,
    RunType,
    TwoThetaBins,
)


def focus_data(
    data: MaskedData[RunType],
    dspacing_bins: DspacingBins,
    twotheta_bins: Optional[TwoThetaBins] = None,
) -> FocussedData[RunType]:
    bins = {}
    if twotheta_bins is not None:
        bins["two_theta"] = twotheta_bins
    bins[dspacing_bins.dim] = dspacing_bins

    if (twotheta_bins is None) or ("two_theta" in data.bins.coords):
        data = data.bins.concat()

    return FocussedData[RunType](data.bin(**bins))


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


providers = (finalize_histogram, focus_data)
"""Sciline providers for grouping pixels."""
