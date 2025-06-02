# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Grouping and merging of pixels / voxels."""

import numpy as np
import scipp as sc

from .types import (
    CountsDspacing,
    DspacingBins,
    FocussedDataDspacing,
    FocussedDataDspacingTwoTheta,
    KeepEvents,
    ReducedCountsDspacing,
    RunType,
    ScaledCountsDspacing,
    TwoThetaBins,
)


def focus_data_dspacing_and_two_theta(
    data: CountsDspacing[RunType],
    dspacing_bins: DspacingBins,
    keep_events: KeepEvents[RunType],
) -> ReducedCountsDspacing[RunType]:
    """
    Reduce the pixel-based data to d-spacing and two-theta dimensions.

    The two-theta binning does not use :py:class:`TwoThetaBins` but instead
    computes the two-theta bins from the 'two_theta' coordinate of the input data. This
    is necessary to ensure that we have sufficiently high wavelength resolution when
    performing a monitor normalization in a follow-up workflow step. If we were to use
    :py:class:`TwoThetaBins` we would be influenced by and limited to the tw-theta
    binning the user requests for the end result, which may not be sufficient.

    Parameters
    ----------
    data:
        The input data to be reduced, which must have 'wavelength', 'dspacing',
        'two_theta' coordinates.
    dspacing_bins:
        The bins to use for the d-spacing dimension.
    keep_events:
        Whether to keep the events in the output. If `False`, the output will be
        histogrammed instead of binned.
    Returns
    -------
    :
        The reduced data with 'dspacing' and 'two_theta' dimensions.
    """
    ttheta = data.coords['two_theta']
    ttheta_min = ttheta.nanmin()
    ttheta_max = ttheta.nanmax()
    ttheta_max.value = np.nextafter(ttheta_max.value, np.inf)
    twotheta_bins = sc.linspace(
        'two_theta',
        start=ttheta_min,
        stop=ttheta_max,
        num=1024,
        unit=ttheta.unit,
    )
    args = {twotheta_bins.dim: twotheta_bins, dspacing_bins.dim: dspacing_bins}
    if keep_events.value:
        result = data.bin(args)
    else:
        # It would be cheaper to simply use `result = data.hist(args)` and computing
        # wavelength from bin centers. This would however not result in a consistent
        # wavelength, unless we do so using the d-spacing calibration table.
        stripped = data.bins.drop_coords(
            list(set(data.bins.coords) - {'dspacing', 'wavelength'})
        )
        binned = stripped.bin(args)
        result = binned.hist()
        result.coords['wavelength'] = binned.bins.coords['wavelength'].bins.nanmean()
    return ReducedCountsDspacing[RunType](result)


def integrate_two_theta(
    data: ScaledCountsDspacing[RunType],
) -> FocussedDataDspacing[RunType]:
    """Integrate the two-theta dimension of the data."""
    if 'two_theta' not in data.dims:
        raise ValueError("Data does not have a 'two_theta' dimension.")
    return FocussedDataDspacing[RunType](
        data.nansum(dim='two_theta')
        if data.bins is None
        else data.bins.concat('two_theta')
    )


def group_two_theta(
    data: ScaledCountsDspacing[RunType],
    two_theta_bins: TwoThetaBins,
) -> FocussedDataDspacingTwoTheta[RunType]:
    """Group the data by two-theta bins."""
    if 'two_theta' not in data.dims:
        raise ValueError("Data does not have a 'two_theta' dimension.")
    data = data.assign_coords(two_theta=sc.midpoints(data.coords['two_theta']))
    return FocussedDataDspacingTwoTheta[RunType](
        data.groupby('two_theta', bins=two_theta_bins).nansum('two_theta')
        if data.bins is None
        else data.bin(two_theta=two_theta_bins)
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


providers = (
    focus_data_dspacing_and_two_theta,
    integrate_two_theta,
    group_two_theta,
)
"""Sciline providers for grouping pixels."""
