# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Grouping and merging of pixels / voxels."""

from .types import (
    DspacingBins,
    FocussedDataDspacing,
    FocussedDataDspacingTwoTheta,
    MaskedData,
    RunType,
    TwoThetaBins,
)


def focus_data_dspacing(
    data: MaskedData[RunType],
    dspacing_bins: DspacingBins,
) -> FocussedDataDspacing[RunType]:
    out = data.bins.concat().bin({dspacing_bins.dim: dspacing_bins})
    return FocussedDataDspacing[RunType](out)


def focus_data_dspacing_and_two_theta(
    data: MaskedData[RunType],
    dspacing_bins: DspacingBins,
    twotheta_bins: TwoThetaBins,
) -> FocussedDataDspacingTwoTheta[RunType]:
    bins = {twotheta_bins.dim: twotheta_bins, dspacing_bins.dim: dspacing_bins}
    if "two_theta" in data.bins.coords:
        data = data.bins.concat()
    return FocussedDataDspacingTwoTheta[RunType](data.bin(**bins))


providers = (focus_data_dspacing, focus_data_dspacing_and_two_theta)
"""Sciline providers for grouping pixels."""
