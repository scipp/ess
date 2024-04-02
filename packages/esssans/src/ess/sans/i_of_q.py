# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import uuid
from typing import Optional

import sciline
import scipp as sc
from scipp.scipy.interpolate import interp1d

from .common import mask_range
from .logging import get_logger
from .types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    CleanDirectBeam,
    CleanMonitor,
    CleanQ,
    CleanSummedQ,
    CleanSummedQMergedBanks,
    DimsToKeep,
    DirectBeam,
    Filename,
    FinalSummedQ,
    IofQ,
    IofQPart,
    MonitorType,
    NeXusDetectorName,
    NonBackgroundWavelengthRange,
    QBins,
    QxyBins,
    ReturnEvents,
    RunType,
    SampleRun,
    ScatteringRunType,
    UncertaintyBroadcastMode,
    WavelengthBins,
    WavelengthMonitor,
)
from .uncertainty import broadcast_with_upper_bound_variances


def preprocess_monitor_data(
    monitor: WavelengthMonitor[RunType, MonitorType],
    wavelength_bins: WavelengthBins,
    non_background_range: Optional[NonBackgroundWavelengthRange],
    uncertainties: UncertaintyBroadcastMode,
) -> CleanMonitor[RunType, MonitorType]:
    """
    Prepare monitor data for computing the transmission fraction.
    The input data are first converted to wavelength (if needed).
    If a ``non_background_range`` is provided, it defines the region where data is
    considered not to be background, and regions outside are background. A mean
    background level will be computed from the background and will be subtracted from
    the non-background counts.
    Finally, if wavelength bins are provided, the data is rebinned to match the
    requested binning.

    Parameters
    ----------
    monitor:
        The monitor to be pre-processed.
    wavelength_bins:
        The binning in wavelength to use for the rebinning.
    non_background_range:
        The range of wavelengths that defines the data which does not constitute
        background. Everything outside this range is treated as background counts.
    uncertainties:
        The mode for broadcasting uncertainties. See
        :py:class:`UncertaintyBroadcastMode` for details.

    Returns
    -------
    :
        The input monitors converted to wavelength, cleaned of background counts, and
        rebinned to the requested wavelength binning.
    """
    background = None
    if non_background_range is not None:
        mask = sc.DataArray(
            data=sc.array(dims=[non_background_range.dim], values=[True]),
            coords={non_background_range.dim: non_background_range},
        )
        background = mask_range(monitor, mask=mask).mean()

    if monitor.bins is not None:
        monitor = monitor.hist(wavelength=wavelength_bins)
    else:
        monitor = monitor.rebin(wavelength=wavelength_bins)

    if background is not None:
        if uncertainties == UncertaintyBroadcastMode.drop:
            monitor -= sc.values(background)
        elif uncertainties == UncertaintyBroadcastMode.upper_bound:
            monitor -= broadcast_with_upper_bound_variances(
                background, sizes=monitor.sizes
            )
        else:
            monitor -= background
    return CleanMonitor(monitor)


def resample_direct_beam(
    direct_beam: DirectBeam, wavelength_bins: WavelengthBins
) -> CleanDirectBeam:
    """
    If the wavelength binning of the direct beam function does not match the requested
    ``wavelength_bins``, perform a 1d interpolation of the function onto the bins.

    Parameters
    ----------
    direct_beam:
        The DataArray containing the direct beam function (it should have a dimension
        of wavelength).
    wavelength_bins:
        The binning in wavelength that the direct beam function should be resampled to.

    Returns
    -------
    :
        The direct beam function resampled to the requested resolution.
    """
    if sc.identical(direct_beam.coords['wavelength'], wavelength_bins):
        return direct_beam
    if direct_beam.variances is not None:
        logger = get_logger('sans')
        logger.warning(
            'An interpolation is being performed on the direct_beam function. '
            'The variances in the direct_beam function will be dropped.'
        )
    func = interp1d(
        sc.values(direct_beam),
        'wavelength',
        fill_value="extrapolate",
        bounds_error=False,
    )
    return CleanDirectBeam(func(wavelength_bins, midpoints=True))


def bin_in_q(
    data: CleanQ[ScatteringRunType, IofQPart],
    q_bins: Optional[QBins],
    qxy_bins: Optional[QxyBins],
    dims_to_keep: Optional[DimsToKeep],
) -> CleanSummedQ[ScatteringRunType, IofQPart]:
    """
    Merges data from all pixels into a single I(Q) spectrum:

    * In the case of event data, events in all bins are concatenated
    * In the case of dense data, counts in all spectra are summed

    Parameters
    ----------
    data:
        A DataArray containing the data that is to be converted to Q.
    q_bins:
        The binning in Q to be used.
    dims_to_keep:
        Dimensions that should not be reduced and thus still be present in the final
        I(Q) result (this is typically the layer dimension).

    Returns
    -------
    :
        The input data converted to Q and then summed over all detector pixels.
    """
    dims_to_reduce = set(data.dims) - {'wavelength'}
    if dims_to_keep is not None:
        dims_to_reduce -= set(dims_to_keep)

    if qxy_bins:
        # We make Qx the inner dim, such that plots naturally show Qx on the x-axis.
        edges = {'Qy': qxy_bins['Qy'], 'Qx': qxy_bins['Qx']}
    else:
        edges = {'Q': q_bins}

    if data.bins is not None:
        q_all_pixels = data.bins.concat(dims_to_reduce)
        # q_all_pixels may just have a single bin now, which currently yields
        # inferior performance when binning (no/bad multi-threading?).
        # We operate on the content buffer for better multi-threaded performance.
        if q_all_pixels.ndim == 0:
            content = q_all_pixels.bins.constituents['data']
            out = content.bin(**edges).assign_coords(q_all_pixels.coords)
        else:
            out = q_all_pixels.bin(**edges)
    else:
        # We want to flatten data to make histogramming cheaper (avoiding allocation of
        # large output before summing). We strip unnecessary content since it makes
        # flattening more expensive.
        stripped = data.copy(deep=False)
        for name, coord in data.coords.items():
            if (
                name not in {'Q', 'Qx', 'Qy', 'wavelength'}
                and set(coord.dims) & dims_to_reduce
            ):
                del stripped.coords[name]
        to_flatten = [dim for dim in data.dims if dim in dims_to_reduce]

        # Make dims to flatten contiguous, keep wavelength as the last dim
        data_dims = list(stripped.dims)
        for dim in to_flatten + ['wavelength']:
            data_dims.remove(dim)
            data_dims.append(dim)
        stripped = stripped.transpose(data_dims)
        # Flatten to helper dim such that `hist` will turn this into the new Q dim(s).
        # For sc.hist this has to be named 'Q'.
        helper_dim = 'Q'
        flat = stripped.flatten(dims=to_flatten, to=helper_dim)

        if len(edges) == 1:
            out = flat.hist(**edges)
        else:
            # sc.hist (or the underlying sc.bin) cannot deal with extra data dims,
            # work around by flattening and regrouping.
            for dim in flat.dims:
                if dim == helper_dim:
                    continue
                if dim not in flat.coords:
                    flat.coords[dim] = sc.arange(dim, flat.sizes[dim])
            out = (
                flat.flatten(to=str(uuid.uuid4()))
                .group(*[flat.coords[dim] for dim in flat.dims if dim != helper_dim])
                .drop_coords(dims_to_keep or ())
                .hist(**edges)
            )
    return CleanSummedQ[ScatteringRunType, IofQPart](out.squeeze())


def no_bank_merge(
    data: CleanSummedQ[ScatteringRunType, IofQPart]
) -> CleanSummedQMergedBanks[ScatteringRunType, IofQPart]:
    return CleanSummedQMergedBanks[ScatteringRunType, IofQPart](data)


def no_run_merge(
    data: CleanSummedQMergedBanks[ScatteringRunType, IofQPart]
) -> FinalSummedQ[ScatteringRunType, IofQPart]:
    return FinalSummedQ[ScatteringRunType, IofQPart](data)


def _merge_contributions(data: list[sc.DataArray]) -> sc.DataArray:
    if len(data) == 1:
        return data[0]
    reducer = sc.reduce(data)
    return reducer.bins.concat() if data[0].bins is not None else reducer.sum()


def merge_banks(
    banks: sciline.Series[NeXusDetectorName, CleanSummedQ[ScatteringRunType, IofQPart]]
) -> CleanSummedQMergedBanks[ScatteringRunType, IofQPart]:
    """
    Merge the events or counts from multiple detector banks into a single numerator or
    denominator, before the normalization step.
    """
    return CleanSummedQMergedBanks[ScatteringRunType, IofQPart](
        _merge_contributions(list(banks.values()))
    )


def merge_runs(
    runs: sciline.Series[
        Filename[ScatteringRunType],
        CleanSummedQMergedBanks[ScatteringRunType, IofQPart],
    ],
) -> FinalSummedQ[ScatteringRunType, IofQPart]:
    """
    Merge the events or counts from multiple runs into a single numerator or
    denominator, before the normalization step.
    """
    return FinalSummedQ[ScatteringRunType, IofQPart](
        _merge_contributions(list(runs.values()))
    )


def subtract_background(
    sample: IofQ[SampleRun],
    background: IofQ[BackgroundRun],
    return_events: ReturnEvents,
) -> BackgroundSubtractedIofQ:
    if return_events and sample.bins is not None and background.bins is not None:
        return sample.bins.concatenate(-background)
    if sample.bins is not None:
        sample = sample.bins.sum()
    if background.bins is not None:
        background = background.bins.sum()
    return BackgroundSubtractedIofQ(sample - background)


providers = (
    preprocess_monitor_data,
    resample_direct_beam,
    bin_in_q,
    subtract_background,
    no_bank_merge,
    no_run_merge,
)
