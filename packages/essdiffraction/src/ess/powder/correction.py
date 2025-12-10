# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Correction algorithms for powder diffraction."""

import enum
from typing import TypeVar

import sciline
import scipp as sc

import ess.reduce
from ess.reduce.uncertainty import broadcast_uncertainties

from ._util import event_or_outer_coord
from .types import (
    AccumulatedProtonCharge,
    CaveMonitor,
    CorrectedDspacing,
    EmptyCanRun,
    EmptyCanSubtractedIofDspacing,
    EmptyCanSubtractedIofDspacingTwoTheta,
    FocussedDataDspacing,
    FocussedDataDspacingTwoTheta,
    IntensityDspacing,
    IntensityDspacingTwoTheta,
    NormalizedDspacing,
    RunType,
    SampleRun,
    UncertaintyBroadcastMode,
    VanadiumRun,
    WavelengthMonitor,
)


def normalize_by_monitor_histogram(
    detector: CorrectedDspacing[RunType],
    *,
    monitor: WavelengthMonitor[RunType, CaveMonitor],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> NormalizedDspacing[RunType]:
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
    detector, skip_range_check = _mask_out_of_monitor_range_data(detector, monitor)
    return NormalizedDspacing[RunType](
        ess.reduce.normalization.normalize_by_monitor_histogram(
            detector=detector,
            monitor=monitor,
            uncertainty_broadcast_mode=uncertainty_broadcast_mode,
            skip_range_check=skip_range_check,
        )
    )


def normalize_by_monitor_integrated(
    detector: CorrectedDspacing[RunType],
    *,
    monitor: WavelengthMonitor[RunType, CaveMonitor],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> NormalizedDspacing[RunType]:
    """Normalize detector data by an integrated monitor.

    The detector is normalized according to

    .. math::

        d_i^\\text{Norm} = \\frac{d_i}{\\sum_j\\, m_j}

    Note that this is not a true integral but only a sum over monitor events.

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
    ess.reduce.normalization.normalize_by_monitor_integrated:
        For details and the actual implementation.
    """
    detector, skip_range_check = _mask_out_of_monitor_range_data(detector, monitor)
    return NormalizedDspacing[RunType](
        ess.reduce.normalization.normalize_by_monitor_integrated(
            detector=detector,
            monitor=monitor,
            uncertainty_broadcast_mode=uncertainty_broadcast_mode,
            skip_range_check=skip_range_check,
        )
    )


def _mask_out_of_monitor_range_data(
    detector: sc.DataArray, monitor: sc.DataArray
) -> tuple[sc.DataArray, bool]:
    if (coord := detector.coords.get("wavelength")) is not None:
        if detector.bins is None and "wavelength" not in coord.dims:
            # The detector was histogrammed early, and the wavelength was reconstructed
            # from d-spacing and 2theta, see focus_data_dspacing_and_two_theta.
            # This introduces unphysical wavelength bins that we need to mask.
            #
            # The detector wavelength coord is bin-edges in d-spacing, but we need
            # the mask to not be bin-edges, so compute the mask on the left and
            # right d-spacing edges and `or` them together.
            mon_coord = monitor.coords["wavelength"]
            mon_lo, mon_hi = mon_coord.min(), mon_coord.max()
            left, right = coord['dspacing', :-1], coord['dspacing', 1:]
            out_of_range = left < mon_lo
            out_of_range |= left > mon_hi
            out_of_range |= right < mon_lo
            out_of_range |= right > mon_hi
            if sc.any(out_of_range):
                return detector.assign_masks(out_of_wavelength_range=out_of_range), True
    return detector, False


def _normalize_by_vanadium(
    data: sc.DataArray,
    vanadium: sc.DataArray,
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> sc.DataArray:
    norm = (
        vanadium.hist(data.coords)
        if vanadium.is_binned
        else vanadium.rebin(data.coords)
    )
    norm = broadcast_uncertainties(
        norm, prototype=data, mode=uncertainty_broadcast_mode
    )
    # Convert the unit such that the end-result has unit 'one' because the division
    # might otherwise produce a unit with a large scale if the proton charges in data
    # and vanadium were measured with different units.
    norm = norm.to(unit=data.unit, copy=False)
    normed = data / norm
    mask = norm.data == sc.scalar(0.0, unit=norm.unit)
    if mask.any():
        normed.masks['zero_vanadium'] = mask
    return normed


_RunTypeNoVanadium = TypeVar("_RunTypeNoVanadium", SampleRun, EmptyCanRun)


def normalize_by_vanadium_dspacing(
    data: FocussedDataDspacing[_RunTypeNoVanadium],
    vanadium: FocussedDataDspacing[VanadiumRun],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> IntensityDspacing[_RunTypeNoVanadium]:
    """Normalize sample data binned in d-spacing by a vanadium measurement.

    If the vanadium data is binned, it gets :func:`histogrammed <scipp.hist>` to the
    same bins as ``data``. If it is not binned, it gets :func:`rebinned <scipp.rebin>`
    to the same coordinates as ``data``. Then, the result is computed as

    .. code-block:: python

        data / vanadium

    And any bins where vanadium is zero are masked out
    with a mask called "zero_vanadium".

    Parameters
    ----------
    data:
        Sample data.
    vanadium:
        Vanadium data.
    uncertainty_broadcast_mode:
        Choose how uncertainties of vanadium are broadcast to the sample data.
        Defaults to ``UncertaintyBroadcastMode.fail``.

    Returns
    -------
    :
        ``data / vanadium``.
        May contain a mask "zero_vanadium" which is ``True``
        for bins where vanadium is zero.

    See Also
    --------
    normalize_by_vanadium_dspacing_and_two_theta:
        Normalization for 2d data binned in d-spacing and :math`2\\theta`.
    """
    return IntensityDspacing(
        _normalize_by_vanadium(data, vanadium, uncertainty_broadcast_mode)
    )


def normalize_by_vanadium_dspacing_and_two_theta(
    data: FocussedDataDspacingTwoTheta[_RunTypeNoVanadium],
    vanadium: FocussedDataDspacingTwoTheta[VanadiumRun],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> IntensityDspacingTwoTheta[_RunTypeNoVanadium]:
    """Normalize sample data binned in (d-spacing, 2theta) by a vanadium measurement.

    If the vanadium data is binned, it gets :func:`histogrammed <scipp.hist>` to the
    same bins as ``data``. If it is not binned, it gets :func:`rebinned <scipp.rebin>`
    to the same coordinates as ``data``. Then, the result is computed as

    .. code-block:: python

        data / vanadium

    And any bins where vanadium is zero are masked out
    with a mask called "zero_vanadium".

    Parameters
    ----------
    data:
        Sample data.
    vanadium:
        Vanadium data.
    uncertainty_broadcast_mode:
        Choose how uncertainties of vanadium are broadcast to the sample data.
        Defaults to ``UncertaintyBroadcastMode.fail``.

    Returns
    -------
    :
        ``data / vanadium``.
        May contain a mask "zero_vanadium" which is ``True``
        for bins where vanadium is zero.

    See Also
    --------
    normalize_by_vanadium_dspacing:
        Normalization for 1d data binned in d-spacing.
    """
    return IntensityDspacingTwoTheta(
        _normalize_by_vanadium(data, vanadium, uncertainty_broadcast_mode)
    )


def normalize_by_proton_charge(
    data: CorrectedDspacing[RunType],
    proton_charge: AccumulatedProtonCharge[RunType],
) -> NormalizedDspacing[RunType]:
    """Normalize data by an accumulated proton charge.

    Parameters
    ----------
    data:
        Un-normalized data array as events or a histogram.
    proton_charge:
        Accumulated proton charge over the entire run.

    Returns
    -------
    :
        ``data / proton_charge``
    """
    return NormalizedDspacing[RunType](data / proton_charge)


def merge_calibration(*, into: sc.DataArray, calibration: sc.Dataset) -> sc.DataArray:
    """
    Return a scipp.DataArray containing calibration metadata as coordinates.

    Parameters
    ----------
    into:
        Base data and metadata for the returned object.
    calibration:
        Calibration parameters.

    Returns
    -------
    :
        Copy of `into` with additional coordinates and masks
        from `calibration`.

    See Also
    --------
    ess.snspowder.powgen.calibration.load_calibration
    """
    for name, coord in calibration.coords.items():
        if not sc.identical(into.coords[name], coord):
            raise ValueError(
                f"Coordinate {name} of calibration and target dataset do not agree."
            )
    out = into.copy(deep=False)
    for name in ("difa", "difc", "tzero"):
        if name in out.coords:
            raise ValueError(
                f"Cannot add calibration parameter '{name}' to data, "
                "there already is metadata with the same name."
            )
        out.coords[name] = calibration[name].data
    if "calibration" in out.masks:
        raise ValueError(
            "Cannot add calibration mask 'calibration' tp data, "
            "there already is a mask with the same name."
        )
    out.masks["calibration"] = calibration["mask"].data
    return out


def apply_lorentz_correction(da: sc.DataArray) -> sc.DataArray:
    """Perform a Lorentz correction for ToF powder diffraction data.

    This function uses this definition:

    .. math::

        L = d^4 \\sin\\theta

    where :math:`d` is d-spacing, :math:`\\theta` is half the scattering angle
    (note the definitions in
    https://scipp.github.io/scippneutron/user-guide/coordinate-transformations.html).

    The Lorentz factor as defined here is suitable for correcting time-of-flight data
    expressed in wavelength or d-spacing.
    It follows the definition used by GSAS-II, see page 140 of
    https://subversion.xray.aps.anl.gov/EXPGUI/gsas/all/GSAS%20Manual.pdf

    Parameters
    ----------
    da:
        Input data with coordinates ``two_theta`` and ``dspacing``.

    Returns
    -------
    :
        ``da`` multiplied by :math:`L`.
        Has the same dtype as ``da``.
    """
    dspacing = event_or_outer_coord(da, "dspacing")
    two_theta = event_or_outer_coord(da, "two_theta")
    sin_theta = sc.sin(0.5 * two_theta)
    d4 = dspacing**4
    out = da * sin_theta.to(dtype=da.bins.dtype if da.bins else da.dtype, copy=False)
    out *= d4.to(dtype=da.bins.dtype if da.bins else da.dtype, copy=False)
    return out


def subtract_empty_can(
    data: IntensityDspacing[SampleRun],
    background: IntensityDspacing[EmptyCanRun],
) -> EmptyCanSubtractedIofDspacing[SampleRun]:
    return EmptyCanSubtractedIofDspacing(data.bins.concatenate(-background))


def subtract_empty_can_two_theta(
    data: IntensityDspacingTwoTheta[SampleRun],
    background: IntensityDspacingTwoTheta[EmptyCanRun],
) -> EmptyCanSubtractedIofDspacingTwoTheta[SampleRun]:
    return EmptyCanSubtractedIofDspacingTwoTheta(data.bins.concatenate(-background))


class RunNormalization(enum.Enum):
    """Type of normalization applied to each run."""

    monitor_histogram = enum.auto()
    monitor_integrated = enum.auto()
    proton_charge = enum.auto()


def insert_run_normalization(
    workflow: sciline.Pipeline, run_norm: RunNormalization
) -> None:
    """Insert providers for a specific normalization into a workflow."""
    match run_norm:
        case RunNormalization.monitor_histogram:
            workflow.insert(normalize_by_monitor_histogram)
        case RunNormalization.monitor_integrated:
            workflow.insert(normalize_by_monitor_integrated)
        case RunNormalization.proton_charge:
            workflow.insert(normalize_by_proton_charge)


providers = (
    normalize_by_proton_charge,
    normalize_by_vanadium_dspacing,
    normalize_by_vanadium_dspacing_and_two_theta,
    subtract_empty_can,
    subtract_empty_can_two_theta,
)
"""Sciline providers for powder diffraction corrections."""
