# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

from typing import Tuple
import scipp as sc
from scipp.constants import g
from .common import gravity_vector
from . import conversions
from . import normalization
from ..wfm.tools import to_bin_centers, to_bin_edges
from scipp.interpolate import interp1d


def _make_coordinate_transform_graphs(gravity):
    """
    Create unit conversion graphs
    """
    data_graph = conversions.sans_elastic(gravity=gravity)
    monitor_graph = conversions.sans_monitor()
    return data_graph, monitor_graph


def _convert_to_wavelength(data: sc.DataArray, monitors: dict, data_graph: dict,
                           monitor_graph: dict) -> Tuple[sc.DataArray, dict]:
    """
    Convert the data array and all the items inside the dict of monitors to wavelength.
    The gravity parameter can be used to turn on or off the effects of gravity.
    """

    # if gravity:
    #     # Make shallow copy of data so as to not add coords in-place
    #     data = data.copy(deep=False)
    #     data.coords["gravity"] = gravity_vector()

    # # Create unit conversion graphs
    # graph = conversions.sans_elastic(gravity=gravity)
    # graph_monitor = conversions.sans_monitor()

    # Convert to wavelength
    data = data.transform_coords("wavelength", graph=data_graph)
    monitors = {
        key: monitors[key].transform_coords("wavelength", graph=monitor_graph)
        for key in monitors
    }

    return data, monitors


def _denoise_and_rebin_monitors(monitors: dict, wavelength_bins: sc.Variable,
                                non_background_range: sc.Variable) -> dict:
    """
    Subtract a background baseline from monitor counts, taken as the mean of the counts
    outside the specified ``non_background_range``.
    """
    return {
        key: normalization.subtract_background_and_rebin(
            monitors[key],
            wavelength_bins=wavelength_bins,
            non_background_range=non_background_range)
        for key in monitors
    }


def _direct_beam(direct_beam: sc.DataArray,
                 wavelength_bins: sc.Variable) -> sc.DataArray:
    """
    If the wavelength binning of the direct beam function does not match the requested
    ``wavelength_bins``, perform a 1d interpolation of the function onto the bins.
    """
    if sc.identical(direct_beam.coords['wavelength'], wavelength_bins):
        return direct_beam
    func = interp1d(sc.values(direct_beam), 'wavelength')
    direct_beam = func(wavelength_bins, midpoints=True)
    logger = sc.get_logger()
    logger.warning('An interpolation was performed on the direct_beam function. '
                   'The variances in the direct_beam function have been ignored.')
    return direct_beam


def _denominator(direct_beam: sc.DataArray, data_incident_monitor: sc.DataArray,
                 transmission_fraction: sc.DataArray,
                 solid_angle: sc.Variable) -> sc.DataArray:
    """
    Compute the denominator term.
    Because we are histogramming the Q values of the denominator further down in the
    workflow, we convert the wavelength coordinate of the denominator from bin edges to
    bin centers.
    """
    denominator = (solid_angle * direct_beam * data_incident_monitor *
                   transmission_fraction)
    # TODO: once scipp-0.12 is released, use sc.midpoints()
    denominator.coords['wavelength'] = 0.5 * (denominator.coords['wavelength'][1:] +
                                              denominator.coords['wavelength'][:-1])
    return denominator


def _convert_to_q_and_merge_spectra(data, graph, wavelength_bands, gravity):
    """
    Convert the data to momentum vector Q. This accepts both dense and event data.
    The final step merges all spectra:
      - In the case of event data, events in all bins are concatenated
      - In the case of dense data, counts in all spectra are summed
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
    if wavelength_bands.sizes['wavelength'] == 2:
        out = out['wavelength', 0]
    return out


def _convert_events_to_q_and_merge_spectra(data, graph, q_bins, wavelength_bands):
    data = data.transform_coords("Q", graph=graph)
    q_boundaries = sc.concat([q_bins.min(), q_bins.max()], dim='Q')

    # TODO: once scipp-0.12 is out, we no longer need to move the attr into the coords
    data.bins.coords['wavelength'] = data.bins.attrs.pop('wavelength')

    q_binned = sc.bin(data, edges=[wavelength_bands, q_boundaries])
    q_summed = q_binned.bins.concat('spectrum')
    return q_summed


def _convert_dense_to_q_and_merge_spectra(data, graph, q_bins, wavelength_bands):
    bands = []
    print(wavelength_bands.sizes['wavelength'])
    for i in range(wavelength_bands.sizes['wavelength'] - 1):
        band = data['wavelength', wavelength_bands[i]:wavelength_bands[i + 1]]
        q_band = band.transform_coords("Q", graph=graph)
        bands.append(sc.histogram(q_band, bins=q_bins))
        # bands.append(sc.histogram(q_band, bins=q_bins).sum('spectrum'))
    q_summed = sc.concat(bands, 'wavelength').sum('spectrum')
    # q_summed = sc.concat(denominator_bands, 'wavelength')
    return q_summed


def _normalize(numerator, denominator, dim='Q'):
    if numerator.bins is not None:
        return numerator.bins / sc.lookup(func=denominator, dim=dim)
    else:
        return numerator / denominator


def q1d(data,
        data_incident_monitor,
        data_transmission_monitor,
        direct_incident_monitor,
        direct_transmission_monitor,
        direct_beam,
        wavelength_bins,
        q_bins,
        gravity=False,
        monitor_non_background_range=None,
        wavelength_bands=None):

    monitors = {
        'data_incident_monitor': data_incident_monitor,
        'data_transmission_monitor': data_transmission_monitor,
        'direct_incident_monitor': direct_incident_monitor,
        'direct_transmission_monitor': direct_transmission_monitor
    }

    data_graph, monitor_graph = _make_coordinate_transform_graphs(gravity=gravity)

    data, monitors = _convert_to_wavelength(data=data,
                                            monitors=monitors,
                                            data_graph=data_graph,
                                            monitor_graph=monitor_graph)

    monitors = _denoise_and_rebin_monitors(monitors=monitors,
                                           wavelength_bins=wavelength_bins,
                                           non_background_range=non_background_range)

    transmission_fraction = normalization.transmission_fraction(**monitors)

    solid_angle = normalization.solid_angle_of_rectangular_pixels(
        data,
        pixel_width=data.coords['pixel_width'],
        pixel_height=data.coords['pixel_height'])

    direct_beam = _direct_beam(direct_beam=direct_beam, wavelength_bins=wavelength_bins)

    denominator = _denominator(direct_beam=direct_beam,
                               data_incident_monitor=monitors['data_incident_monitor'],
                               transmission_fraction=transmission_fraction,
                               solid_angle=solid_angle)
    # Insert copy coords needed for conversion to Q.
    # TODO: can this be avoided by copying the Q coords from the converted numerator?
    for coord in ['position', 'sample_position', 'source_position']:
        denominator.coords[coord] = data.meta[coord]

    # 3. Convert from wavelength to Q ==================================================

    # In the case where no wavelength bands are requested, we create a single wavelength
    # band to make sure we select the correct wavelength range that corresponds to
    # wavelength_bins
    if wavelength_bands is None:
        wavelength_bands = sc.concat(
            [wavelength_bins.min(), wavelength_bins.max()], dim='wavelength')

    data_q = _convert_to_q_and_merge_spectra(data=data,
                                             graph=data_graph,
                                             wavelength_bands=wavelength_bands,
                                             gravity=gravity)

    denominator_q = _convert_to_q_and_merge_spectra(data=denominator,
                                                    graph=data_graph,
                                                    wavelength_bands=wavelength_bands,
                                                    gravity=gravity)

    # data = data.transform_coords("Q", graph=graph)
    # q_boundaries = sc.concat([q_bins.min(), q_bins.max()], dim='Q')

    # # TODO: once scipp-0.12 is out, we no longer need to move the attr into the coords
    # data.bins.coords['wavelength'] = data.bins.attrs.pop('wavelength')

    # q_binned = sc.bin(data, edges=[wavelength_bands, q_boundaries])
    # q_summed = q_binned.bins.concat('spectrum')

    # denominator_bands = []
    # for i in range(number_of_wavelength_bands):
    #     band = denominator['wavelength',
    #                        wavelength_bands['wavelength',
    #                                         i]:wavelength_bands['wavelength', i + 1]]
    #     q_band = band.transform_coords("Q", graph=graph)
    #     denominator_bands.append(sc.histogram(q_band, bins=q_bins))
    # denominator_q_summed = sc.concat(denominator_bands, 'wavelength').sum('spectrum')

    # 4. Normalize =====================================================================

    # normalized = q_summed.bins / sc.lookup(func=denominator_q_summed, dim='Q')
    normalized = _normalize(numerator=data_q, denominator=denominator_q, dim='Q')
    # if histogram_output:
    #     normalized = sc.histogram(normalized, bins=q_bins)
    # if number_of_wavelength_bands == 1:
    #     normalized = normalized['wavelength', 0]
    return normalized
