# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Grouping and merging of pixels / voxels."""

from .types import (
    DspacingBins,
    DspacingData,
    FocussedDataDspacing,
    FocussedDataDspacingTwoTheta,
    RunType,
    TwoThetaBins,
)


def focus_data_dspacing(
    data: DspacingData[RunType], dspacing_bins: DspacingBins
) -> FocussedDataDspacing[RunType]:
    return FocussedDataDspacing[RunType](
        data.bin({dspacing_bins.dim: dspacing_bins}, dim=data.dims)
    )


def focus_data_dspacing_and_two_theta(
    data: DspacingData[RunType],
    dspacing_bins: DspacingBins,
    twotheta_bins: TwoThetaBins,
) -> FocussedDataDspacingTwoTheta[RunType]:
    return FocussedDataDspacingTwoTheta[RunType](
        data.bin({twotheta_bins.dim: twotheta_bins, dspacing_bins.dim: dspacing_bins})
    )


providers = (focus_data_dspacing, focus_data_dspacing_and_two_theta)
"""Sciline providers for grouping pixels."""
