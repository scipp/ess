# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.scipy.interpolate import interp1d

from ess.reduce.uncertainty import UncertaintyBroadcastMode, broadcast_uncertainties

from .common import mask_range
from .logging import get_logger
from .types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BackgroundSubtractedIofQxy,
    CleanDirectBeam,
    QDetector,
    QxyDetector,
    BinnedQ,
    BinnedQxQy,
    CorrectedMonitor,
    DimsToKeep,
    DirectBeam,
    IntensityQ,
    IntensityQxQy,
    IofQPart,
    MonitorType,
    NonBackgroundWavelengthRange,
    QBins,
    QxBins,
    QyBins,
    ReturnEvents,
    RunType,
    SampleRun,
    ScatteringRunType,
    WavelengthBins,
    WavelengthMonitor,
)


def preprocess_monitor_data(
    monitor: WavelengthMonitor[RunType, MonitorType],
    wavelength_bins: WavelengthBins,
    non_background_range: NonBackgroundWavelengthRange,
    uncertainties: UncertaintyBroadcastMode,
) -> CorrectedMonitor[RunType, MonitorType]:
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
        :py:class:`ess.reduce.uncertainty.UncertaintyBroadcastMode` for details.

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
        monitor -= broadcast_uncertainties(
            background, prototype=monitor, mode=uncertainties
        )
    return CorrectedMonitor(monitor)


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
    if direct_beam is None:
        return CleanDirectBeam(
            sc.DataArray(
                sc.ones(dims=wavelength_bins.dims, shape=[len(wavelength_bins) - 1]),
                coords={'wavelength': wavelength_bins},
            )
        )
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
    data: QDetector[ScatteringRunType, IofQPart],
    q_bins: QBins,
    dims_to_keep: DimsToKeep,
) -> BinnedQ[ScatteringRunType, IofQPart]:
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
    out = _bin_in_q(data=data, edges={'Q': q_bins}, dims_to_keep=dims_to_keep)
    return BinnedQ[ScatteringRunType, IofQPart](out)


def bin_in_qxy(
    data: QxyDetector[ScatteringRunType, IofQPart],
    qx_bins: QxBins,
    qy_bins: QyBins,
    dims_to_keep: DimsToKeep,
) -> BinnedQxQy[ScatteringRunType, IofQPart]:
    """
    Merges data from all pixels into a single I(Q) spectrum:

    * In the case of event data, events in all bins are concatenated
    * In the case of dense data, counts in all spectra are summed

    Parameters
    ----------
    data:
        A DataArray containing the data that is to be converted to Q.
    qx_bins:
        The binning in Qx to be used.
    qy_bins:
        The binning in Qy to be used.
    dims_to_keep:
        Dimensions that should not be reduced and thus still be present in the final
        I(Q) result (this is typically the layer dimension).

    Returns
    -------
    :
        The input data converted to Qx and Qy and then summed over all detector pixels.
    """
    # We make Qx the inner dim, such that plots naturally show Qx on the x-axis.
    out = _bin_in_q(
        data=data,
        edges={'Qy': qy_bins, 'Qx': qx_bins},
        dims_to_keep=dims_to_keep,
    )
    return BinnedQxQy[ScatteringRunType, IofQPart](out)


def _bin_in_q(
    data: sc.DataArray, edges: dict[str, sc.Variable], dims_to_keep: tuple[str, ...]
) -> sc.DataArray:
    dims_to_reduce = set(data.dims) - {'wavelength'} - set(dims_to_keep or ())
    return (data.hist if data.bins is None else data.bin)(**edges, dim=dims_to_reduce)


def _subtract_background(
    sample: sc.DataArray,
    background: sc.DataArray,
    return_events: ReturnEvents,
) -> sc.DataArray:
    if return_events and sample.bins is not None and background.bins is not None:
        return sample.bins.concatenate(-background)
    if sample.bins is not None:
        sample = sample.bins.sum()
    if background.bins is not None:
        background = background.bins.sum()
    return sample - background


def subtract_background(
    sample: IntensityQ[SampleRun],
    background: IntensityQ[BackgroundRun],
    return_events: ReturnEvents,
) -> BackgroundSubtractedIofQ:
    return BackgroundSubtractedIofQ(
        _subtract_background(
            sample=sample, background=background, return_events=return_events
        )
    )


def subtract_background_xy(
    sample: IntensityQxQy[SampleRun],
    background: IntensityQxQy[BackgroundRun],
    return_events: ReturnEvents,
) -> BackgroundSubtractedIofQxy:
    return BackgroundSubtractedIofQxy(
        _subtract_background(
            sample=sample, background=background, return_events=return_events
        )
    )


providers = (
    preprocess_monitor_data,
    resample_direct_beam,
    bin_in_q,
    bin_in_qxy,
    subtract_background,
    subtract_background_xy,
)
