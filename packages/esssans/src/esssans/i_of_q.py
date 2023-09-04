# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Dict, Optional, Union

import scipp as sc
from scipp.scipy.interpolate import interp1d

from .logging import get_logger
from . import conversions, normalization
from .common import gravity_vector, mask_range
from .types import (
    DirectBeam,
    CleanDirectBeam,
    WavelengthBins,
    WavelengthMonitor,
    RunType,
    MonitorType,
    CleanMonitor,
    WavelengthBins,
    NonBackgroundWavelengthRange,
)


def preprocess_monitor_data(
    monitor: WavelengthMonitor[RunType, MonitorType],
    wavelength_bins: WavelengthBins,
    non_background_range: NonBackgroundWavelengthRange,
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

    Returns
    -------
    :
        The input monitors converted to wavelength, cleaned of background counts, and
        rebinned to the requested wavelength binning.
    """
    monitor = monitor.value
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
        # TODO: reference Heybrock et al. (2023) paper
        # For subtracting the background from the monitors, we need to remove the
        # variances because the broadcasting operation will fail.
        # We add a simple check comparing the background level to the total number
        # of counts.
        # TODO: is this check good enough? See https://github.com/scipp/ess/issues/174
        bg = sc.values(background)
        if (bg / monitor.sum()).value > 0.1:
            raise ValueError(
                'The background level is more than 10% of the total monitor counts. '
                'Dropping the variances of the background would drop non-negligible '
                'contributions to uncertainties from correlations.'
            )
        monitor = monitor - bg
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


def convert_to_q_and_merge_spectra(
    data: sc.DataArray,
    graph: dict,
    q_bins: Union[int, sc.Variable],
    gravity: bool,
    wavelength_bands: Optional[sc.Variable] = None,
) -> sc.DataArray:
    """
    Convert the data to momentum vector Q. This accepts both dense and event data.
    The final step merges all spectra:

    * In the case of event data, events in all bins are concatenated
    * In the case of dense data, counts in all spectra are summed

    Parameters
    ----------
    data:
        A DataArray containing the data that is to be converted to Q.
    graph:
        The coordinate conversion graph used to perform the conversion to Q.
    q_bins:
        The binning in Q to be used.
    gravity:
        If ``True``, include the effects of gravity when computing the scattering angle.
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
    if gravity and ('gravity' not in data.meta):
        data = data.copy(deep=False)
        data.coords["gravity"] = gravity_vector()

    if data.bins is not None:
        out = _convert_events_to_q_and_merge_spectra(
            data=data, graph=graph, q_bins=q_bins, wavelength_bands=wavelength_bands
        )
    else:
        out = _convert_dense_to_q_and_merge_spectra(
            data=data, graph=graph, q_bins=q_bins, wavelength_bands=wavelength_bands
        )
    if (wavelength_bands is not None) and (wavelength_bands.sizes['wavelength'] == 2):
        out = out['wavelength', 0]
    return out


def _to_q_bins(q_bins: Union[int, sc.Variable]) -> Dict[str, Union[int, sc.Variable]]:
    """
    If the input bins are an integer, convert them to a dictionary that can be used
    to bin a DataArray.
    """
    if isinstance(q_bins, int):
        return {'Q': q_bins}
    return {q_bins.dim: q_bins}


def _convert_events_to_q_and_merge_spectra(
    data: sc.DataArray,
    graph: dict,
    q_bins: Union[int, sc.Variable],
    wavelength_bands: Optional[sc.Variable] = None,
) -> sc.DataArray:
    """
    Convert event data to momentum vector Q.
    """
    data_q = data.transform_coords('Q', graph=graph)
    q_all_pixels = data_q.bins.concat(set(data_q.dims) - {'Q'})
    edges = _to_q_bins(q_bins)
    if wavelength_bands is not None:
        edges[wavelength_bands.dim] = wavelength_bands
    return q_all_pixels.bin(**edges)


def _convert_dense_to_q_and_merge_spectra(
    data: sc.DataArray,
    graph: dict,
    q_bins: Union[int, sc.Variable],
    wavelength_bands: Optional[sc.Variable] = None,
) -> sc.DataArray:
    """
    Convert dense data to momentum vector Q.
    """
    bands = []
    data_q = data.transform_coords('Q', graph=graph)
    sum_dims = set(data_q.dims) - {'Q'}
    edges = _to_q_bins(q_bins)
    if wavelength_bands is None:
        return data_q.hist(**edges).sum(sum_dims)
    for i in range(wavelength_bands.sizes['wavelength'] - 1):
        band = data_q['wavelength', wavelength_bands[i] : wavelength_bands[i + 1]]
        bands.append(band.hist(**edges).sum(sum_dims))
    q_summed = sc.concat(bands, 'wavelength')
    return q_summed


def to_I_of_Q(
    data: sc.DataArray,
    data_monitors: Dict[str, sc.DataArray],
    direct_monitors: Dict[str, sc.DataArray],
    direct_beam: sc.DataArray,
    wavelength_bins: sc.Variable,
    q_bins: Union[int, sc.Variable],
    gravity: bool = False,
    wavelength_mask: Optional[sc.DataArray] = None,
    wavelength_bands: Optional[sc.Variable] = None,
    signal_over_monitor_threshold: float = 0.1,
) -> sc.DataArray:
    """
    Compute the scattering cross-section I(Q) for a SANS experimental run, performing
    binning in Q and a normalization based on monitor data and a direct beam function.

    The main steps of the workflow are:

    * Generate a coordinate transformation graph from ``tof`` to ``Q``, that also
      includes ``wavelength``.
    * Convert the detector data to wavelength.
    * Compute the transmission fraction from the monitor data.
    * Compute the solid angles of the detector pixels.
    * Resample the direct beam function to the same wavelength binning as the
      monitors.
    * Combine solid angle, direct beam, transmission fraction and incident monitor
      counts to compute the denominator for the normalization.
    * Convert the detector data to momentum vector Q.
    * Convert the denominator to momentum vector Q.
    * Normalize the detector data.

    Parameters
    ----------
    data:
        The detector data. This can be both events or dense (histogrammed) data.
    data_monitors:
        The data arrays for the incident and transmission monitors for the measurement
        run (background noise removed and converted to wavelength).
    direct_monitors:
        The data arrays for the incident and transmission monitors for the direct
        run (background noise removed and converted to wavelength).
    direct_beam:
        The direct beam function of the instrument (histogrammed,
        depends on wavelength).
    wavelength_bins:
        The binning in the wavelength dimension to be used.
    q_bins:
        The binning in the Q dimension to be used.
    gravity:
        Include the effects of gravity when computing the scattering angle if ``True``.
    wavelength_mask:
        Mask to apply to the wavelength coordinate (to mask out artifacts from the
        instrument beamline). See :func:`common.mask_range` for more details.
    wavelength_bands:
        If defined, return the data as a set of bands in the wavelength dimension. This
        is useful for separating different wavelength ranges that contribute to
        different regions in Q space.
    signal_over_monitor_threshold:
        The threshold for the ratio of detector counts to monitor counts above which
        an error is raised because it is not safe to drop the variances of the monitor.

    Returns
    -------
    :
        The intensity as a function of Q.
    """

    # Convert sample data to wavelength
    graph = conversions.sans_elastic(gravity=gravity)
    data = data.transform_coords("wavelength", graph=graph)

    if wavelength_mask is not None:
        # If we have binned data and the wavelength coord is multi-dimensional, we need
        # to make a single wavelength bin before we can mask the range.
        if data.bins is not None:
            dim = wavelength_mask.dim
            if (dim in data.bins.coords) and (dim in data.coords):
                data = data.bin({dim: 1})
        data = mask_range(data, wavelength_mask)
        data_monitors = {
            key: mask_range(mon, wavelength_mask) for key, mon in data_monitors.items()
        }
        direct_monitors = {
            key: mask_range(mon, wavelength_mask)
            for key, mon in direct_monitors.items()
        }

    # Compute normalizing term
    direct_beam = resample_direct_beam(
        direct_beam=direct_beam, wavelength_bins=wavelength_bins
    )
    denominator = normalization.iofq_denominator(
        data=data,
        data_transmission_monitor=data_monitors['transmission'],
        direct_incident_monitor=direct_monitors['incident'],
        direct_transmission_monitor=direct_monitors['transmission'],
        direct_beam=direct_beam,
        signal_over_monitor_threshold=signal_over_monitor_threshold,
    )

    # Insert a copy of coords needed for conversion to Q.
    # TODO: can this be avoided by copying the Q coords from the converted numerator?
    for coord in ['position', 'sample_position', 'source_position']:
        denominator.coords[coord] = data.meta[coord]

    # In the case where no wavelength bands are requested, we create a single wavelength
    # band to make sure we select the correct wavelength range that corresponds to
    # wavelength_bins
    if wavelength_bands is None:
        wavelength_bands = sc.concat(
            [wavelength_bins.min(), wavelength_bins.max()], dim='wavelength'
        )

    data_q = convert_to_q_and_merge_spectra(
        data=data,
        graph=graph,
        wavelength_bands=wavelength_bands,
        q_bins=q_bins,
        gravity=gravity,
    )

    denominator_q = convert_to_q_and_merge_spectra(
        data=denominator,
        graph=graph,
        wavelength_bands=wavelength_bands,
        q_bins=q_bins,
        gravity=gravity,
    )

    normalized = normalization.normalize(numerator=data_q, denominator=denominator_q)

    return normalized


providers = [preprocess_monitor_data]
