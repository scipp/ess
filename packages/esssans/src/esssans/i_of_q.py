# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Dict, Optional, Union

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
    DirectBeam,
    IofQ,
    IofQPart,
    MonitorType,
    NonBackgroundWavelengthRange,
    QBins,
    RunType,
    SampleRun,
    UncertaintyBroadcastMode,
    WavelengthBands,
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
    func = interp1d(sc.values(direct_beam), 'wavelength')
    direct_beam = func(wavelength_bins, midpoints=True)
    logger = get_logger('sans')
    logger.warning(
        'An interpolation was performed on the direct_beam function. '
        'The variances in the direct_beam function have been dropped.'
    )
    return CleanDirectBeam(direct_beam)


def merge_spectra(
    data: CleanQ[RunType, IofQPart],
    q_bins: QBins,
    wavelength_bands: WavelengthBands,
) -> CleanSummedQ[RunType, IofQPart]:
    """
    Merges all spectra:

    * In the case of event data, events in all bins are concatenated
    * In the case of dense data, counts in all spectra are summed

    Parameters
    ----------
    data:
        A DataArray containing the data that is to be converted to Q.
    q_bins:
        The binning in Q to be used.
    wavelength_bands:
        Defines bands in wavelength that can be used to separate different wavelength
        ranges that contribute to different regions in Q space. Note that this needs to
        be defined, so if all wavelengths should be used, this should simply be a start
        and end edges that encompass the entire wavelength range.

    Returns
    -------
    :
        The input data converted to Q and then summed over all detector pixels.
    """
    if data.bins is not None:
        out = _events_merge_spectra(
            data_q=data, q_bins=q_bins, wavelength_bands=wavelength_bands
        )
    else:
        out = _dense_merge_spectra(
            data_q=data, q_bins=q_bins, wavelength_bands=wavelength_bands
        )
    if (wavelength_bands is not None) and (wavelength_bands.sizes['wavelength'] == 2):
        out = out['wavelength', 0]
    return CleanSummedQ[RunType, IofQPart](out)


def _to_q_bins(q_bins: Union[int, sc.Variable]) -> Dict[str, Union[int, sc.Variable]]:
    """
    If the input bins are an integer, convert them to a dictionary that can be used
    to bin a DataArray.
    """
    if isinstance(q_bins, int):
        return {'Q': q_bins}
    return {q_bins.dim: q_bins}


def _events_merge_spectra(
    data_q: sc.DataArray,
    q_bins: Union[int, sc.Variable],
    wavelength_bands: Optional[sc.Variable] = None,
) -> sc.DataArray:
    """
    Merge spectra of event data
    """
    q_all_pixels = data_q.bins.concat(set(data_q.dims) - {'Q'})
    edges = _to_q_bins(q_bins)
    if wavelength_bands is not None:
        edges[wavelength_bands.dim] = wavelength_bands
    return q_all_pixels.bin(**edges)


def _dense_merge_spectra(
    data_q: sc.DataArray,
    q_bins: Union[int, sc.Variable],
    wavelength_bands: Optional[sc.Variable] = None,
) -> sc.DataArray:
    """
    Merge spectra of dense data
    """
    bands = []
    sum_dims = set(data_q.dims) - {'Q'}
    edges = _to_q_bins(q_bins)
    if wavelength_bands is None:
        return data_q.hist(**edges).sum(sum_dims)
    for i in range(wavelength_bands.sizes['wavelength'] - 1):
        band = data_q['wavelength', wavelength_bands[i] : wavelength_bands[i + 1]]
        bands.append(band.hist(**edges).sum(sum_dims))
    q_summed = sc.concat(bands, 'wavelength')
    return q_summed


def subtract_background(
    sample: IofQ[SampleRun], background: IofQ[BackgroundRun]
) -> BackgroundSubtractedIofQ:
    if sample.bins is not None:
        sample = sample.bins.sum()
    if background.bins is not None:
        background = background.bins.sum()
    return BackgroundSubtractedIofQ(sample - background)


providers = [
    preprocess_monitor_data,
    resample_direct_beam,
    merge_spectra,
    subtract_background,
]
