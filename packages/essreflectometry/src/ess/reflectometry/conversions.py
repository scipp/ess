# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import pi
from scippneutron._utils import elem_dtype


def reflectometry_q(wavelength: sc.Variable, theta: sc.Variable) -> sc.Variable:
    """
    Compute momentum transfer from reflection angle.

    Parameters
    ----------
    wavelength:
        Wavelength values for the events.
    theta:
        Angle of reflection for the events.

    Returns
    -------
    :
        Q-values.
    """
    dtype = elem_dtype(wavelength)
    c = (4 * pi).astype(dtype)
    return c * sc.sin(theta.astype(dtype, copy=False)) / wavelength


def reflectometry_q_x(
    wavelength: sc.Variable,
    incident_angle: sc.Variable,
    reflection_angle: sc.Variable,
) -> sc.Variable:
    """Compute coplanar momentum transfer parallel to the sample surface."""
    dtype = elem_dtype(wavelength)
    c = (2 * pi).astype(dtype)
    return (
        c
        * (
            sc.cos(reflection_angle.astype(dtype, copy=False))
            - sc.cos(incident_angle.to(unit=reflection_angle.unit, dtype=dtype))
        )
        / wavelength
    )


def reflectometry_q_z(
    wavelength: sc.Variable,
    incident_angle: sc.Variable,
    reflection_angle: sc.Variable,
) -> sc.Variable:
    """Compute coplanar momentum transfer normal to the sample surface."""
    dtype = elem_dtype(wavelength)
    c = (2 * pi).astype(dtype)
    return (
        c
        * (
            sc.sin(reflection_angle.astype(dtype, copy=False))
            + sc.sin(incident_angle.to(unit=reflection_angle.unit, dtype=dtype))
        )
        / wavelength
    )


providers = ()
