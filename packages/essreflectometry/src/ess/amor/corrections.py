# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.types import ProtonCharge, RunType
from .types import (
    ChopperFrequency,
    ProtonCurrent,
    ProtonCurrentThreshold,
    PulseTimes,
)


def proton_charge_per_pulse(
    current: ProtonCurrent[RunType],
    pulse_times: PulseTimes[RunType],
    chopper_frequency: ChopperFrequency[RunType],
    threshold: ProtonCurrentThreshold,
) -> ProtonCharge[RunType]:
    """Proton charge on the target during each chopper revolution.

    The charge is the proton current at the start of the revolution
    multiplied by the duration of the revolution.
    Revolutions where the current does not exceed the threshold are masked,
    the neutrons recorded in them are discarded.

    Returns
    -------
    :
        The proton charge as a function of the start time of the revolution.
    """
    current_at = sc.lookup(
        current,
        'time',
        mode='previous',
        fill_value=sc.zeros_like(threshold),
    )
    current_per_pulse = current_at(pulse_times)
    return ProtonCharge[RunType](
        sc.DataArray(
            (current_per_pulse / chopper_frequency).to(unit='uC'),
            coords={'time': pulse_times},
            masks={'proton_current_too_low': ~(current_per_pulse > threshold)},
        )
    )


def mask_events_where_proton_current_is_too_low(
    da: sc.DataArray, proton_charge: sc.DataArray
) -> sc.DataArray:
    """Mask the events of the chopper revolutions that are masked in the proton charge.

    Parameters
    ----------
    da:
        Events with the event coordinate ``event_time_zero``.
    proton_charge:
        Proton charge of each chopper revolution, see :func:`proton_charge_per_pulse`.

    Returns
    -------
    :
        The events with the added event mask ``proton_current_too_low``.
    """
    charge_at = sc.lookup(
        proton_charge,
        'time',
        mode='previous',
        fill_value=sc.scalar(float('nan'), unit=proton_charge.unit),
    )
    # The lookup fills masked charge entries with NaN. Transfer the existing
    # pulse selection to events; the current threshold is only evaluated above.
    return da.bins.assign_masks(
        proton_current_too_low=sc.isnan(charge_at(da.bins.coords['event_time_zero']))
    )


def normalize_by_proton_charge(
    da: sc.DataArray, proton_charge: sc.DataArray
) -> sc.DataArray:
    """Divide the data by the total proton charge of the accepted chopper revolutions.

    In contrast to :func:`ess.reflectometry.corrections.correct_by_proton_charge`
    all events get the same weight. Revolutions without events contribute to the
    charge, and the result is a rate that can be compared between runs.
    """
    return da / proton_charge.sum().data


providers = (proton_charge_per_pulse,)
