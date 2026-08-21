# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Default providers and parameters for the SKADI SANS workflow."""

import sciline
import scipp as sc
import scippnexus as snx
from ess import sans
from ess.sans.parameters import typical_outputs

from ess.reduce.uncertainty import UncertaintyBroadcastMode
from ess.reduce.unwrap import WavelengthLutMode
from ess.reduce.workflow import register_workflow

from ..sans.types import (
    DetectorMasks,
    DirectBeam,
    MonitorTerm,
    Position,
    RawDetector,
    ReturnEvents,
    RunType,
    SolidAngle,
    WavelengthBins,
)
from .mcstas import mcstas_providers


def skadi_default_parameters() -> dict:
    """Return defaults for a minimal SKADI reduction."""
    return {
        DetectorMasks: {},
        DirectBeam: None,
        ReturnEvents: False,
        UncertaintyBroadcastMode: UncertaintyBroadcastMode.drop,
    }


def rectangular_pixel_solid_angle(
    detector: RawDetector[RunType],
    sample_position: Position[snx.NXsample, RunType],
) -> SolidAngle[RunType]:
    """Compute the solid angle of flat rectangular SKADI pixels.

    The detector data must contain ``position``, ``pixel_size``, and
    ``detector_normal`` coordinates. This requirement is independent of the source
    data format; a NeXus loader can supply the same calibrated coordinates as the
    McStas loader.
    """
    missing = {
        'position',
        'pixel_size',
        'detector_normal',
    } - set(detector.coords)
    if missing:
        raise ValueError(
            "SKADI detector data is missing geometry coordinates: "
            + ', '.join(sorted(missing))
        )

    scattered_beam = detector.coords['position'] - sample_position
    distance = sc.norm(scattered_beam)
    area = (
        detector.coords['pixel_size'].fields.x * detector.coords['pixel_size'].fields.y
    )
    projected_area = (
        area
        * sc.abs(sc.dot(detector.coords['detector_normal'], scattered_beam))
        / distance
    )
    omega = projected_area / distance**2

    coords = {
        name: coord
        for name, coord in detector.coords.items()
        if set(coord.dims).issubset(detector.dims)
    }
    return SolidAngle[RunType](sc.DataArray(omega, coords=coords))


def unity_monitor_term(wavelength_bins: WavelengthBins) -> MonitorTerm[RunType]:
    """Return unity incident-flux and transmission normalization.

    This makes the basic workflow usable for simulations without monitor data. The
    standard SANS solid-angle normalization and optional direct-beam correction remain
    in the workflow. Replace this provider when measured monitor and transmission data
    are available.
    """
    wavelength = sc.midpoints(wavelength_bins)
    return MonitorTerm[RunType](
        sc.DataArray(sc.ones(sizes=wavelength.sizes), coords={'wavelength': wavelength})
    )


skadi_providers = (rectangular_pixel_solid_angle, unity_monitor_term)


@register_workflow
def SkadiWorkflow(
    wavelength_from: WavelengthLutMode = "file",
) -> sciline.Pipeline:
    """Create a basic, data-source-independent SKADI reduction workflow.

    Parameters
    ----------
    wavelength_from:
        Mode used by the common SANS workflow to obtain wavelength. A data-source
        adapter may override this conversion, as :func:`SkadiMcStasWorkflow` does.

    Returns
    -------
    :
        The SKADI reduction workflow.
    """
    workflow = sans.SansWorkflow(wavelength_from=wavelength_from)
    for provider in skadi_providers:
        workflow.insert(provider)
    for key, value in skadi_default_parameters().items():
        workflow[key] = value
    workflow.typical_outputs = typical_outputs
    return workflow


@register_workflow
def SkadiMcStasWorkflow() -> sciline.Pipeline:
    """Create the basic SKADI workflow with the McStas input adapter."""
    workflow = SkadiWorkflow()
    for provider in mcstas_providers:
        workflow.insert(provider)
    return workflow
