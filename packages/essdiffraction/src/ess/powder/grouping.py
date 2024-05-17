# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Grouping and merging of pixels / voxels."""
from typing import Optional

from scippneutron.conversion.graph import beamline

from .types import (
    DetectorDimensions,
    DspacingBins,
    DspacingData,
    DspacingHistogram,
    FocussedData,
    MaskedData,
    NormalizedByVanadium,
    RunType,
    TwoThetaBins,
)


def focus_data(
    data: MaskedData[RunType],
    detector_dims: DetectorDimensions,
    dspacing_bins: DspacingBins,
    twotheta_bins: Optional[TwoThetaBins] = None,
) -> FocussedData[RunType]:
    bins = {}
    if twotheta_bins is not None:
        bins["two_theta"] = twotheta_bins
    bins[dspacing_bins.dim] = dspacing_bins

    if twotheta_bins is None:
        # In this case merge data from all pixels
        # Put the dims into the same order as in the data.
        # See https://github.com/scipp/scipp/issues/3408
        to_concat = tuple(dim for dim in data.dims if dim in detector_dims)
        data = data.bins.concat(to_concat)

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
