# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Grouping and merging of pixels / voxels."""

import scipp as sc

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


def collect_detectors(*detectors: sc.DataArray) -> sc.DataGroup:
    """Store all inputs in a single data group.

    This function is intended to be used to reduce a workflow which
    was mapped over detectors.

    Parameters
    ----------
    detectors:
        Data arrays for each detector bank.
        All arrays must have a scalar "detector" coord containing a ``str``.

    Returns
    -------
    :
        The inputs as a data group with the "detector" coord as the key.
    """
    return sc.DataGroup({da.coords.pop('detector').value: da for da in detectors})


providers = (focus_data_dspacing, focus_data_dspacing_and_two_theta)
"""Sciline providers for grouping pixels."""
