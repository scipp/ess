# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import ess.reduce
import scipp as sc
from ess.reduce.uncertainty import UncertaintyBroadcastMode

from ..reflectometry.corrections import correct_by_proton_current
from ..reflectometry.types import (
    BeamDivergenceLimits,
    CoordTransformationGraph,
    CorrectionsToApply,
    ProtonCurrent,
    ReducibleData,
    RunType,
    WavelengthBins,
    WavelengthDetector,
    YIndexLimits,
    ZIndexLimits,
)
from .conversions import add_coords
from .maskings import add_masks
from .types import WavelengthMonitor


def normalize_by_monitor_histogram(
    detector: ReducibleData[RunType],
    *,
    monitor: WavelengthMonitor[RunType],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> ReducibleData[RunType]:
    """Normalize detector data by a histogrammed monitor.

    The detector is normalized according to

    .. math::

        d_i^\\text{Norm} = \\frac{d_i}{m_i} \\Delta \\lambda_i

    Parameters
    ----------
    detector:
        Input event data in wavelength.
    monitor:
        A histogrammed monitor in wavelength.
    uncertainty_broadcast_mode:
        Choose how uncertainties of the monitor are broadcast to the sample data.

    Returns
    -------
    :
        `detector` normalized by a monitor.

    See also
    --------
    ess.reduce.normalization.normalize_by_monitor_histogram:
        For details and the actual implementation.
    """
    return ess.reduce.normalization.normalize_by_monitor_histogram(
        detector=detector,
        monitor=monitor,
        uncertainty_broadcast_mode=uncertainty_broadcast_mode,
        skip_range_check=False,
    )


def add_coords_masks_and_apply_corrections(
    da: WavelengthDetector[RunType],
    ylim: YIndexLimits,
    zlims: ZIndexLimits,
    bdlim: BeamDivergenceLimits,
    wbins: WavelengthBins,
    proton_current: ProtonCurrent[RunType],
    monitor: WavelengthMonitor[RunType],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
    graph: CoordTransformationGraph[RunType],
    corrections_to_apply: CorrectionsToApply,
) -> ReducibleData[RunType]:
    """
    Computes coordinates, masks and corrections that are
    the same for the sample measurement and the reference measurement.
    """
    da = add_coords(da, graph)
    da = add_masks(da, ylim, zlims, bdlim, wbins)

    for correction in corrections_to_apply:
        if correction == 'monitor':
            da = normalize_by_monitor_histogram(
                da,
                monitor=monitor,
                uncertainty_broadcast_mode=uncertainty_broadcast_mode,
            )
        elif correction == 'proton_current':
            da = correct_by_proton_current(da, proton_current=proton_current)
        else:
            da = correction(da)

    return ReducibleData[RunType](da)


def correct_by_footprint(da: sc.DataArray) -> sc.DataArray:
    """Corrects the data by the size of the footprint on the sample."""
    return da / sc.sin(da.coords['theta'])


default_corrections = {correct_by_proton_current, correct_by_footprint}

providers = (add_coords_masks_and_apply_corrections,)
