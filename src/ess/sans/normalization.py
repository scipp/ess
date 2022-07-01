# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippneutron as scn


def solid_angle_of_rectangular_pixels(data: sc.DataArray, pixel_width: sc.Variable,
                                      pixel_height: sc.Variable) -> sc.Variable:
    """
    Solid angle computed from rectangular pixels with a 'width' and a 'height'.

    Note that this is an approximation which is only valid for small angles
    between the line of sight and the rectangle normal.

    :param data: The DataArray that contains the positions for the detector pixels and
        the sample.
    :param pixel_width: The width of the rectangular pixels.
    :param pixel_height: The height of the rectangular pixels.
    """
    L2 = scn.L2(data)
    return (pixel_width * pixel_height) / (L2 * L2)


def transmission_fraction(data_monitors: dict, direct_monitors: dict) -> sc.DataArray:
    """
    Approximation based on equations in CalculateTransmission documentation
    p = \frac{S_T}{D_T}\frac{D_I}{S_I}
    This is equivalent to mantid.CalculateTransmission without fitting.

    TODO: It seems we are always multiplying this by data_monitors['incident'] to
    compute the normalization term. We could consider just returning
    data_monitors['transmission'] * direct_monitors['incident'] /
        direct_monitors['transmission']

    :param data_monitors: A dict containing the DataArrays for the incident and
        transmission monitors for the measurement run (monitor data should depend on
        wavelength).
    :param direct_monitors: A dict containing the DataArrays for the incident and
        transmission monitors for the direct run (monitor data should depend on
        wavelength).
    """
    return (data_monitors['transmission'] / direct_monitors['transmission']) * (
        direct_monitors['incident'] / data_monitors['incident'])


def compute_denominator(direct_beam: sc.DataArray, data_incident_monitor: sc.DataArray,
                        transmission_fraction: sc.DataArray,
                        solid_angle: sc.Variable) -> sc.DataArray:
    """
    Compute the denominator term.
    Because we are histogramming the Q values of the denominator further down in the
    workflow, we convert the wavelength coordinate of the denominator from bin edges to
    bin centers.

    :param direct_beam: The DataArray containing the direct beam function (depends on
        wavelength).
    :param data_incident_monitor: The DataArray containing the incident monitor counts
        from the measurement run (depends on wavelength).
    :param transmission_fraction: The DataArray containing the transmission fraction
        (depends on wavelength).
    :param solid_angle: The solid angle of the detector pixels (depends on detector
        position).
    """
    denominator = (solid_angle * direct_beam * data_incident_monitor *
                   transmission_fraction)
    denominator.coords['wavelength'] = sc.midpoints(denominator.coords['wavelength'])
    return denominator


def normalize(numerator: sc.DataArray, denominator: sc.DataArray) -> sc.DataArray:
    """
    Perform normalization of counts as a fucntion of Q.
    If the numerator contains events, we use the sc.lookup function to perform the
    division.

    :param numerator: The data whose counts will be divided by the denominator. This
        can either be event or dense (histogrammed) data.
    :param denominator: The divisor for the normalization operation. This cannot be
        event data, it must contain histogrammed data.
    """
    if numerator.bins is not None:
        return numerator.bins / sc.lookup(func=denominator, dim='Q')
    else:
        return numerator / denominator
