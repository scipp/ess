# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Dict, Optional, Tuple, Union
import uuid

import scipp as sc
from scipp.scipy.interpolate import interp1d

from . import conversions, normalization
from .common import gravity_vector
from ..logging import get_logger

# def make_coordinate_transform_graphs(gravity: bool,
#                                      monitor: bool = False,
#                                      scatter: bool = True) -> dict:
#     """
#     Create unit conversion graphs.
#     The gravity parameter can be used to turn on or off the effects of gravity.

#     Parameters
#     ----------
#     gravity:
#         If ``True``, the coordinate transformation graph will incorporate the
#         effects of the Earth's gravitational field on the flight path of the neutrons
#         when computing the scattering angle.
#     monitor:
#         Re
#     scatter:
#         If ``True``, make graph for scattering beamlines.

#     Returns
#     -------
#     :
#         Two coordinate transformation graphs: the first for the detector pixels, and
#         the second for the beam monitors.
#     """
#     if monitor:
#         return conversions.sans_monitor()
#     else:
#         return conversions.sans_elastic(gravity=gravity, scatter=scatter)


def convert_to_wavelength(data: sc.DataArray, monitors: dict, data_graph: dict,
                          monitor_graph: dict) -> Tuple[sc.DataArray, dict]:
    """
    Convert the data array and all the items inside the dict of monitors to wavelength
    using a pre-defined conversion graph.

    Parameters
    ----------
    data:
        The data from the measurement that is to be converted to wavelength.
    monitors:
        A dict of monitors. All entries in the dict will be converted to wavelength.
    data_graph:
        The coordinate transformation graph to be used for the data.
    monitor_graph:
        The coordinate transformation graph to be used for the monitors.

    Returns
    -------
    :
        The input ``data`` and ``monitors`` converted to wavelength.
    """
    data = data.transform_coords("wavelength", graph=data_graph)
    monitors = monitors_to_wavelength(monitors=monitors, graph=monitor_graph)
    return data, monitors


def monitors_to_wavelength(monitors: Union[Dict[str, sc.DataArray], sc.DataArray],
                           graph: dict) -> Union[Dict[str, sc.DataArray], sc.DataArray]:
    """
    Recursively convert all monitors in dict.

    Parameters
    ----------
    monitors:
        The monitors whose time-of-flight coordinate is to be converted to wavelength.
    graph:
        The coordinate transformation graph to be used for the monitors.

    Returns
    -------
    :
        The input monitors converted to wavelength.
    """
    if isinstance(monitors, dict):
        return {
            key: monitors_to_wavelength(monitors[key], graph=graph)
            for key in monitors
        }
    else:
        return monitors.transform_coords("wavelength", graph=graph)


def denoise_and_rebin_monitors(
    monitors: Union[Dict[str, sc.DataArray], sc.DataArray],
    wavelength_bins: sc.Variable,
    non_background_range: Optional[sc.Variable] = None
) -> Union[Dict[str, sc.DataArray], sc.DataArray]:
    """
    Subtract a background baseline from monitor counts, taken as the mean of the counts
    outside the specified ``non_background_range``.

    Parameters
    ----------
    monitors:
        A DataArray containing monitor data, or a dict of monitor DataArrays.
        In the case of a dict of monitors, all entries in the dict will be background
        subtracted and rebinned.
    wavelength_bins:
        The binning in wavelength to use for the rebinning.
    non_background_range:
        The range of wavelengths that defines the data which does not constitute
        background. Everything outside this range is treated as background counts.

    Returns
    -------
    :
        The input monitors with background signal subtracted from data counts.
    """
    if isinstance(monitors, dict):
        return {
            key: denoise_and_rebin_monitors(monitors[key],
                                            wavelength_bins=wavelength_bins,
                                            non_background_range=non_background_range)
            for key in monitors
        }
    else:
        if non_background_range is not None:
            dim = non_background_range.dim
            monitors.variances = None  # TODO: Hack to set variances to None
            below = monitors[dim, :non_background_range[0]]
            above = monitors[dim, non_background_range[1]:]
            background = sc.concat([below.data, above.data], dim=dim).mean()
            monitors = monitors - background
        return monitors.rebin(wavelength=wavelength_bins)


def resample_direct_beam(direct_beam: sc.DataArray,
                         wavelength_bins: sc.Variable) -> sc.DataArray:
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
    logger.warning('An interpolation was performed on the direct_beam function. '
                   'The variances in the direct_beam function have been dropped.')
    return direct_beam


def convert_to_q_and_merge_spectra(
        data: sc.DataArray,
        graph: dict,
        q_bins: sc.Variable,
        gravity: bool,
        wavelength_bands: Optional[sc.Variable] = None) -> sc.DataArray:
    """
    Convert the data to momentum vector Q. This accepts both dense and event data.
    The final step merges all spectra:
      - In the case of event data, events in all bins are concatenated
      - In the case of dense data, counts in all spectra are summed

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
    if gravity:
        data = data.copy(deep=False)
        data.coords["gravity"] = gravity_vector()

    if data.bins is not None:
        out = _convert_events_to_q_and_merge_spectra(data=data,
                                                     graph=graph,
                                                     q_bins=q_bins,
                                                     wavelength_bands=wavelength_bands)
    else:
        out = _convert_dense_to_q_and_merge_spectra(data=data,
                                                    graph=graph,
                                                    q_bins=q_bins,
                                                    wavelength_bands=wavelength_bands)
    if (wavelength_bands is not None) and (wavelength_bands.sizes['wavelength'] == 2):
        out = out['wavelength', 0]
    return out


def _convert_events_to_q_and_merge_spectra(
        data: sc.DataArray,
        graph: dict,
        q_bins: sc.Variable,
        wavelength_bands: Optional[sc.Variable] = None) -> sc.DataArray:
    """
    Convert event data to momentum vector Q.
    """
    data_q = data.transform_coords("Q", graph=graph)
    q_summed = data_q.bins.concat('spectrum')
    edges = [q_bins]
    if wavelength_bands is not None:
        edges = [wavelength_bands] + edges
    return sc.binning.make_binned(q_summed, edges=edges)


def _convert_dense_to_q_and_merge_spectra(
        data: sc.DataArray,
        graph: dict,
        q_bins: sc.Variable,
        wavelength_bands: Optional[sc.Variable] = None) -> sc.DataArray:
    """
    Convert dense data to momentum vector Q.
    """
    bands = []
    data_q = data.transform_coords("Q", graph=graph)
    data_q.coords['wavelength'] = data_q.attrs.pop('wavelength')
    if wavelength_bands is None:
        return data_q.hist({q_bins.dim: q_bins}).sum('spectrum')
    for i in range(wavelength_bands.sizes['wavelength'] - 1):
        band = data_q['wavelength', wavelength_bands[i]:wavelength_bands[i + 1]]
        bands.append(band.hist({q_bins.dim: q_bins}).sum('spectrum'))
    q_summed = sc.concat(bands, 'wavelength')
    return q_summed


def _make_dict_of_monitors(data_monitors, direct_monitors):
    """
    Also verify that no entries are missing in the monitors and place them into a
    single dict for convenience.
    """
    for group, monitor_dict in zip(('data', 'direct'),
                                   (data_monitors, direct_monitors)):
        for key in ('incident', 'transmission'):
            if key not in monitor_dict:
                raise KeyError(
                    f'The dict of monitors for the {group} run is missing entry {key}.')
    return {'data': data_monitors, 'direct': direct_monitors}


def _add_mask(da: sc.DataArray, mask: sc.DataArray, name: str) -> sc.DataArray:
    """
    Add wavelength mask to data array. If it contains binned data, use a top-level
    mask. It the data is dense, use lookup to find the mask value for each bin.
    """
    if da.bins is not None:
        da = da.bin({mask.dim: mask.coords[mask.dim]})
        da.masks[name] = mask.data
    else:
        lu = sc.lookup(mask, mask.dim)
        if da.coords.is_edges(mask.dim):
            sampling = sc.midpoints(da.coords[mask.dim])
        else:
            sampling = da.coords[mask.dim]
        da.masks[name] = lu[sampling]
    return da


def add_wavelength_mask(data: sc.DataArray, monitors: dict,
                        mask: sc.DataArray) -> Tuple[sc.DataArray, dict]:
    """
    Add wavelength mask to data and monitors.
    """
    mask_name = uuid.uuid4().hex
    data = _add_mask(data, mask=mask, name=mask_name)
    monitors_out = {}
    for group in monitors:
        monitors_out[group] = {}
        for key in monitors[group]:
            monitors_out[group][key] = _add_mask(monitors[group][key],
                                                 mask=mask,
                                                 name=mask_name)
    return data, monitors_out


def normalization_denominator(
        data: sc.DataArray,
        data_monitors: dict,
        direct_monitors: dict,
        direct_beam: sc.DataArray,
        wavelength_bins: sc.Variable,
        monitor_non_background_range: Optional[sc.Variable] = None,
        wavelength_mask: Optional[sc.DataArray] = None) -> sc.DataArray:
    """
    Compute the normalizing term for the SANS I(Q).
    This is basically:
      solid_angle * direct_beam * data_incident_monitor_counts * transmission_fraction

    Parameters
    ----------
    data:
        The DataArray containing the detector data. This can be both events
        or dense (histogrammed) data.
    data_monitors:
        A dict containing the data array for the incident and
        transmission monitors for the measurement run
    direct_monitors:
        A dict containing the data array for the incident and
        transmission monitors for the direct (empty sample holder) run.
    direct_beam:
        The direct beam function of the instrument (histogrammed,
        depends on wavelength).
    wavelength_bins:
        The binning in the wavelength dimension to be used.
    monitor_non_background_range:
        The range of wavelengths for the monitors that are considered to not be part of
        the background. This is used to compute the background level on each monitor,
        which then gets subtracted from each monitor's counts.
    wavelength_mask:
        Mask to apply to the wavelength coordinate (to mask out artifacts from the
        instrument beamline).

    Returns
    -------
    :
        The normalizing term (denominator) in the SANS I(Q) equation.
    """

    monitors = _make_dict_of_monitors(data_monitors=data_monitors,
                                      direct_monitors=direct_monitors)

    monitor_graph = conversions.sans_monitor()

    monitors = monitors_to_wavelength(monitors, graph=monitor_graph)

    monitors = denoise_and_rebin_monitors(
        monitors=monitors,
        wavelength_bins=wavelength_bins,
        non_background_range=monitor_non_background_range)

    if wavelength_mask is not None:
        data, monitors = add_wavelength_mask(data=data,
                                             monitors=monitors,
                                             mask=wavelength_mask)

    transmission_fraction = normalization.transmission_fraction(
        data_monitors=monitors['data'], direct_monitors=monitors['direct'])

    direct_beam = resample_direct_beam(direct_beam=direct_beam,
                                       wavelength_bins=wavelength_bins)

    solid_angle = normalization.solid_angle_of_rectangular_pixels(
        data,
        pixel_width=data.coords['pixel_width'],
        pixel_height=data.coords['pixel_height'])

    return normalization.compute_denominator(
        direct_beam=direct_beam,
        data_incident_monitor=monitors['data']['incident'],
        transmission_fraction=transmission_fraction,
        solid_angle=solid_angle)


def to_I_of_Q(data: sc.DataArray,
              data_monitors: dict,
              direct_monitors: dict,
              direct_beam: sc.DataArray,
              wavelength_bins: sc.Variable,
              q_bins: sc.Variable,
              gravity: bool = False,
              monitor_non_background_range: Optional[sc.Variable] = None,
              wavelength_mask: Optional[sc.DataArray] = None,
              wavelength_bands: Optional[sc.Variable] = None) -> sc.DataArray:
    """
    Compute the scattering cross-section I(Q) for a SANS experimental run, performing
    binning in Q and a normalization based on monitor data and a direct beam function.

    The main steps of the workflow are:

       * Generate a coordinate transformation graph from ``tof`` to ``Q``, that also
         includes ``wavelength``.
       * Convert the detector data and monitors to wavelength.
       * Remove the background noise from the monitors and align them to a common
         binning axis.
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
        The DataArray containing the detector data. This can be both events
        or dense (histogrammed) data.
    data_monitors:
        A dict containing the data array for the incident and
        transmission monitors for the measurement run
    direct_monitors:
        A dict containing the data array for the incident and
        transmission monitors for the direct (empty sample holder) run.
    direct_beam:
        The direct beam function of the instrument (histogrammed,
        depends on wavelength).
    wavelength_bins:
        The binning in the wavelength dimension to be used.
    q_bins:
        The binning in the Q dimension to be used.
    gravity:
        Include the effects of gravity when computing the scattering angle if ``True``.
    monitor_non_background_range:
        The range of wavelengths for the monitors that are considered to not be part of
        the background. This is used to compute the background level on each monitor,
        which then gets subtracted from each monitor's counts.
    wavelength_mask:
        Mask to apply to the wavelength coordinate (to mask out artifacts from the
        instrument beamline).
    wavelength_bands:
        If defined, return the data as a set of bands in the wavelength dimension. This
        is useful for separating different wavelength ranges that contribute to
        different regions in Q space.

    Returns
    -------
    :
        The intensity as a function of Q.
    """

    graph = conversions.sans_elastic(gravity=gravity)

    # Convert data to wavelength
    data = data.transform_coords("wavelength", graph=graph)

    # monitors = _make_dict_of_monitors(data_monitors=data_monitors,
    #                                   direct_monitors=direct_monitors)

    # data_graph, monitor_graph = make_coordinate_transform_graphs(gravity=gravity)

    # data, monitors = convert_to_wavelength(data=data,
    #                                        monitors=monitors,
    #                                        data_graph=data_graph,
    #                                        monitor_graph=monitor_graph)

    # monitors = denoise_and_rebin_monitors(
    #     monitors=monitors,
    #     wavelength_bins=wavelength_bins,
    #     non_background_range=monitor_non_background_range)

    # if wavelength_mask is not None:
    #     data, monitors = add_wavelength_mask(data=data,
    #                                          monitors=monitors,
    #                                          mask=wavelength_mask)

    # transmission_fraction = normalization.transmission_fraction(
    #     data_monitors=monitors['data'], direct_monitors=monitors['direct'])

    # direct_beam = resample_direct_beam(direct_beam=direct_beam,
    #                                    wavelength_bins=wavelength_bins)

    # solid_angle = normalization.solid_angle_of_rectangular_pixels(
    #     data,
    #     pixel_width=data.coords['pixel_width'],
    #     pixel_height=data.coords['pixel_height'])

    # denominator = normalization.compute_denominator(
    #     direct_beam=direct_beam,
    #     data_incident_monitor=monitors['data']['incident'],
    #     transmission_fraction=transmission_fraction,
    #     solid_angle=solid_angle)

    # Compute normalizing term
    denominator = normalization_denominator(
        data=data,
        data_monitors=data_monitors,
        direct_monitors=direct_monitors,
        direct_beam=direct_beam,
        wavelength_bins=wavelength_bins,
        monitor_non_background_range=monitor_non_background_range,
        wavelength_mask=wavelength_mask)

    # Insert a copy of coords needed for conversion to Q.
    # TODO: can this be avoided by copying the Q coords from the converted numerator?
    for coord in ['position', 'sample_position', 'source_position']:
        denominator.coords[coord] = data.meta[coord]

    # In the case where no wavelength bands are requested, we create a single wavelength
    # band to make sure we select the correct wavelength range that corresponds to
    # wavelength_bins
    if wavelength_bands is None:
        wavelength_bands = sc.concat(
            [wavelength_bins.min(), wavelength_bins.max()], dim='wavelength')

    data_q = convert_to_q_and_merge_spectra(data=data,
                                            graph=graph,
                                            wavelength_bands=wavelength_bands,
                                            q_bins=q_bins,
                                            gravity=gravity)

    denominator_q = convert_to_q_and_merge_spectra(data=denominator,
                                                   graph=graph,
                                                   wavelength_bands=wavelength_bands,
                                                   q_bins=q_bins,
                                                   gravity=gravity)

    normalized = normalization.normalize(numerator=data_q, denominator=denominator_q)

    return normalized
