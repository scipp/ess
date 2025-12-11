# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Normalization routines."""

import scipp as sc

from ess.reduce.uncertainty import broadcast_uncertainties
from ess.spectroscopy.types import (
    FrameMonitor3,
    IncidentEnergyDetector,
    NormalizedIncidentEnergyDetector,
    ProtonCharge,
    RunType,
    UncertaintyBroadcastMode,
    WavelengthMonitor,
)


def normalize_by_monitor_and_proton_charge(
    detector: IncidentEnergyDetector[RunType],
    monitor: WavelengthMonitor[RunType, FrameMonitor3],
    proton_charge: ProtonCharge[RunType],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> NormalizedIncidentEnergyDetector[RunType]:
    """Normalize detector data by a monitor and proton charge.

    This function divides the detector event weights by the distribution of the monitor
    in incident wavelength and by the proton charge. The former accounts for the
    wavelength-dependent transmission of the beam through the primary spectrometer.
    The latter accounts for the measurement time and source strength.

    The detector is normalized according to

    .. math::

        d_e^\\text{Norm} = d_e \\frac{\\Delta x_i}{m_i \\sum_j m_j} \\frac1{C}

    where :math:`d_e` is a detector event, :math:`m_i` is the monitor bin containing
    the incident wavelength of that detector event, and :math:`\\Delta x_i` is the bin
    width of that bin. Finally, :math:`C` is the proton charge.
    The monitor term is chosen to have unit integral and represent a probability
    density of incident neutrons.

    Parameters
    ----------
    detector:
        Detector events to normalize.
    monitor:
        Monitor histogram.
    proton_charge:
        Accumulated proton charge.
        Scalar or per setting (a3 / a4).
    uncertainty_broadcast_mode:
        Choose how uncertainties of the monitor and proton charge
        are broadcast to the sample data.

    Returns
    -------
    :
        The detector events normalized by the monitor and proton charge.
    """
    # We should make this function public or move the normalization to ESSreduce.
    from ess.reduce.normalization import _mask_detector_for_norm

    if 'time' in monitor.dims:
        # Use a combined monitor distribution for all instrument settings.
        monitor = monitor.sum('time')

    detector = _mask_detector_for_norm(detector=detector, monitor=monitor)

    norm = _monitor_distribution(monitor=monitor)
    # Combine monitor and proton charge so we only operate on events once.
    norm *= proton_charge
    norm = broadcast_uncertainties(
        norm, prototype=detector, mode=uncertainty_broadcast_mode
    )
    return NormalizedIncidentEnergyDetector[RunType](
        detector.bins / sc.lookup(norm, dim=monitor.dim)
    )


def _monitor_distribution(
    monitor: sc.DataArray,
) -> sc.DataArray:
    coord = monitor.coords[monitor.dim]
    delta_w = sc.DataArray(coord[1:] - coord[:-1], masks=monitor.masks)
    return monitor / delta_w / sc.values(monitor).sum()


providers = (normalize_by_monitor_and_proton_charge,)
