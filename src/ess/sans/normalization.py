# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Dict

import scipp as sc
import scippneutron as scn


def solid_angle_of_rectangular_pixels(data: sc.DataArray, pixel_width: sc.Variable,
                                      pixel_height: sc.Variable) -> sc.DataArray:
    """
    Solid angle computed from rectangular pixels with a 'width' and a 'height'.

    Note that this is an approximation which is only valid for small angles
    between the line of sight and the rectangle normal.

    Parameters
    ----------
    data:
        The DataArray that contains the positions for the detector pixels and the
        sample.
    pixel_width:
        The width of the rectangular pixels.
    pixel_height:
        The height of the rectangular pixels.

    Returns
    -------
    :
        The solid angle of the detector pixels, as viewed from the sample position.
        Any masks that have a dimension common to the dimensions of the position
        coordinate are retained to the output.
    """
    L2 = scn.L2(data)
    omega = (pixel_width * pixel_height) / (L2 * L2)
    solid_angle = sc.DataArray(data=omega)
    for key, mask in data.masks.items():
        omega_dims = set(omega.dims)
        mask_dims = set(mask.dims)
        if omega_dims.issubset(mask_dims) or mask_dims.issubset(omega_dims):
            solid_angle.masks[key] = mask
    return solid_angle


def transmission_fraction(data_monitors: Dict[str, sc.DataArray],
                          direct_monitors: Dict[str, sc.DataArray]) -> sc.DataArray:
    """
    Approximation based on equations in
    [CalculateTransmission](https://docs.mantidproject.org/v4.0.0/algorithms/CalculateTransmission-v1.html)
    documentation
    p = \frac{S_T}{D_T}\frac{D_I}{S_I}
    This is equivalent to ``mantid.CalculateTransmission`` without fitting.
    Inputs should be wavelength-dependent.

    TODO: It seems we are always multiplying this by data_monitors['incident'] to
    compute the normalization term. We could consider just returning
    data_monitors['transmission'] * direct_monitors['incident'] /
        direct_monitors['transmission']

    Parameters
    ----------
    data_monitors:
        The data arrays for the incident and transmission monitors for the measurement
        run (monitor data should depend on wavelength).
    direct_monitors:
        The data arrays for the incident and transmission monitors for the direct
        run (monitor data should depend on wavelength).

    Returns
    -------
    :
        The transmission fraction computed from the monitor counts.
    """  # noqa: E501
    return (data_monitors['transmission'] / direct_monitors['transmission']) * (
        direct_monitors['incident'] / data_monitors['incident'])


def compute_denominator(direct_beam: sc.DataArray, data_incident_monitor: sc.DataArray,
                        transmission_fraction: sc.DataArray,
                        solid_angle: sc.Variable) -> sc.DataArray:
    """
    Compute the denominator term. This is basically:
      solid_angle * direct_beam * data_incident_monitor_counts * transmission_fraction

    Because we are histogramming the Q values of the denominator further down in the
    workflow, we convert the wavelength coordinate of the denominator from bin edges to
    bin centers.

    Parameters
    ----------
    direct_beam:
        The DataArray containing the direct beam function (depends on wavelength).
    data_incident_monitor:
        The DataArray containing the incident monitor counts from the measurement run
        (depends on wavelength).
    transmission_fraction:
        The DataArray containing the transmission fraction (depends on wavelength).
    solid_angle:
        The solid angle of the detector pixels (depends on detector position).

    Returns
    -------
    :
        The denominator for the SANS I(Q) normalization.
    """
    denominator = (solid_angle * direct_beam * data_incident_monitor *
                   transmission_fraction)
    denominator.coords['wavelength'] = sc.midpoints(denominator.coords['wavelength'])
    return denominator


def normalize(numerator: sc.DataArray, denominator: sc.DataArray) -> sc.DataArray:
    """
    Perform normalization of counts as a function of Q.
    If the numerator contains events, we use the sc.lookup function to perform the
    division.

    Parameters
    ----------
    numerator:
        The data whose counts will be divided by the denominator. This can either be
        event or dense (histogrammed) data.
    denominator:
        The divisor for the normalization operation. This cannot be event data, it must
        contain histogrammed data.

    Returns
    -------
    :
        The input data normalized by the supplied denominator.
    """
    if numerator.bins is not None:
        return numerator.bins / sc.lookup(func=denominator, dim='Q')
    else:
        return numerator / denominator
