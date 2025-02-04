# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import pi
from scippneutron._utils import elem_dtype

from .types import ProtonCurrent, RunType


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


def add_proton_current_coord(
    da: sc.DataArray,
    pc: ProtonCurrent[RunType],
) -> sc.DataArray:
    """Find the proton current value for each event and
    adds it as a coord to the data array."""
    pc_lookup = sc.lookup(
        pc,
        dim='time',
        mode='previous',
        fill_value=sc.scalar(float('nan'), unit=pc.unit),
    )
    # Useful for comparing the proton current to what is typical
    da = da.assign_coords(median_proton_current=sc.median(pc).data)
    da.coords.set_aligned('median_proton_current', False)
    da = da.bins.assign_coords(
        proton_current=pc_lookup(da.bins.coords['event_time_zero'])
    )
    return da


def add_proton_current_mask(da: sc.DataArray) -> sc.DataArray:
    """Masks events where the proton current was too low or where
    the proton current is nan."""
    # Take inverse and use >= because we want to mask nan values
    da = da.bins.assign_masks(
        proton_current_too_low=~(
            da.bins.coords['proton_current'] >= da.coords['median_proton_current'] / 4
        )
    )
    return da


providers = ()
