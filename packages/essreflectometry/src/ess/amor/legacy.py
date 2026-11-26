# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Providers for Amor files written before the 2026 change of the file layout.

These providers make up :func:`ess.amor.AmorWorkflow2025`.
They differ from the current workflow in where quantities are stored in the file,
in how the chopper phase enters the wavelength, and in the normalization by
the proton charge, which is applied event by event.
"""

import numpy as np
import scipp as sc
import scippnexus as snx

from ..reflectometry import supermirror
from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.corrections import correct_by_footprint, correct_by_proton_charge
from ..reflectometry.load import load_nx
from ..reflectometry.types import (
    BeamDivergenceLimits,
    BeamSize,
    CoordTransformationGraph,
    DetectorRotation,
    DetectorSpatialResolution,
    Filename,
    NeXusComponent,
    NeXusDetectorName,
    Position,
    ProtonCharge,
    RawChopper,
    RawDetector,
    RawSampleRotation,
    ReducibleData,
    RunType,
    SampleRotation,
    SampleRotationOffset,
    SampleSize,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)
from .conversions import (
    add_coords,
    add_masks,
    divergence_angle,
    theta,
    theta_no_gravity,
)
from .geometry import Detector
from .load import group_events_by_pixel
from .types import (
    ChopperDistance,
    ChopperFrequency,
    ChopperPhase,
    ChopperSeparation,
    GravityToggle,
)


def load_detector(
    file_path: Filename[RunType], detector_name: NeXusDetectorName
) -> NeXusComponent[snx.NXdetector, RunType]:
    """Load the detector group from a NeXus file."""
    return next(load_nx(file_path, f"NXentry/NXinstrument/{detector_name}"))


def load_events(
    detector: NeXusComponent[snx.NXdetector, RunType],
) -> RawDetector[RunType]:
    """Load and reshape event data from a NeXus detector group."""
    return RawDetector[RunType](group_events_by_pixel(detector["data"]))


def load_amor_ch_phase(ch: RawChopper[RunType]) -> ChopperPhase[RunType]:
    """Extract chopper phase from the chopper group."""
    p = ch["phase"]["value"].coords["average_value"].value
    if getattr(p, "unit", None):
        return p
    raise ValueError("No unit was found for the chopper phase")


def load_amor_ch_frequency(ch: RawChopper[RunType]) -> ChopperFrequency[RunType]:
    """Extract chopper frequency from the chopper group."""
    f = ch["rotation_speed"]["value"].coords["average_value"]
    if getattr(f, "unit", None):
        return f
    raise ValueError("No unit was found for the chopper frequency")


def load_amor_sample_rotation(fp: Filename[RunType]) -> RawSampleRotation[RunType]:
    """Load sample rotation log and return the first value."""
    (mu,) = load_nx(fp, "NXentry/NXinstrument/master_parameters/mu")
    # Jochens Amor code reads the first value of this log
    # see https://github.com/jochenstahn/amor/blob/140e3192ddb7e7f28acee87e2acaee65ce1332aa/libeos/file_reader.py#L272
    # might have to change if this field ever becomes truly time-dependent
    return sc.scalar(mu['value'].data['dim_1', 0]['time', 0].value, unit='deg')


def load_amor_detector_rotation(fp: Filename[RunType]) -> DetectorRotation[RunType]:
    """Load detector rotation log and return the first value."""
    (nu,) = load_nx(fp, "NXentry/NXinstrument/master_parameters/nu")
    # Jochens Amor code reads the first value of this log
    # see https://github.com/jochenstahn/amor/blob/140e3192ddb7e7f28acee87e2acaee65ce1332aa/libeos/file_reader.py#L272
    # might have to change if this field ever becomes truly time-dependent
    return sc.scalar(nu['value'].data['dim_1', 0]['time', 0].value, unit='deg')


def load_amor_proton_charge(
    fp: Filename[RunType],
) -> ProtonCharge[RunType]:
    """Load proton current log from the NeXus detector group."""
    (pc,) = load_nx(fp, 'NXentry/NXinstrument/NXdetector/proton_current')
    pc = pc['value']['dim_1', 0]
    pc.data.unit = 'mA/s'
    return pc


def wavelength(
    event_time_offset, pixel_divergence_angle, L1, L2, chopper_phase, chopper_frequency
) -> sc.Variable:
    "Converts event_time_offset to wavelength using the chopper settings."
    out = event_time_offset.to(unit="ns", dtype="float64", copy=True)
    unit = out.bins.unit
    tau = (1 / (2 * chopper_frequency.to(unit='Hz'))).to(unit=unit)
    tof_offset = tau * chopper_phase.to(unit='rad') / (np.pi * sc.units.rad)

    minimum = -tof_offset
    frame_bound = tau - tof_offset
    maximum = 2 * tau - tof_offset

    # Frame unwrapping
    out += sc.where(
        (minimum < event_time_offset) & (event_time_offset < frame_bound),
        tof_offset,
        sc.where(
            (frame_bound < event_time_offset) & (event_time_offset < maximum),
            tof_offset - tau,
            sc.scalar(np.nan, unit=unit),
        ),
    )
    # Correction for path length through guides being different
    # depending on incident angle.
    out -= (pixel_divergence_angle.to(unit="rad") / (np.pi * sc.units.rad)) * tau
    out *= (sc.constants.h / sc.constants.m_n) / (L1 + L2)
    return out.to(unit='angstrom', copy=False)


def coordinate_transformation_graph(
    gravity: GravityToggle,
    detector_rotation: DetectorRotation[RunType],
    sample_rotation: SampleRotation[RunType],
    chopper_phase: ChopperPhase[RunType],
    chopper_frequency: ChopperFrequency[RunType],
    chopper_distance: ChopperDistance[RunType],
    chopper_separation: ChopperSeparation[RunType],
    sample_size: SampleSize[RunType],
    beam_size: BeamSize[RunType],
) -> CoordTransformationGraph[RunType]:
    """Build the coordinate transformation graph for Amor."""
    return {
        "wavelength": wavelength,
        "theta": theta if gravity else theta_no_gravity,
        "divergence_angle": divergence_angle,
        "Q": reflectometry_q,
        "L1": lambda chopper_distance: sc.abs(chopper_distance),
        "L2": lambda distance_in_detector: distance_in_detector + Detector.distance,
        "sample_rotation": lambda: sample_rotation.to(unit='rad'),
        "detector_rotation": lambda: detector_rotation.to(unit='rad'),
        "chopper_phase": lambda: chopper_phase,
        "chopper_frequency": lambda: chopper_frequency,
        "chopper_separation": lambda: chopper_separation,
        "chopper_distance": lambda: chopper_distance,
        "sample_size": lambda: sample_size,
        "beam_size": lambda: beam_size,
    }


def add_coords_masks_and_apply_corrections(
    da: RawDetector[RunType],
    ylim: YIndexLimits,
    zlims: ZIndexLimits,
    bdlim: BeamDivergenceLimits,
    wbins: WavelengthBins,
    proton_charge: ProtonCharge[RunType],
    graph: CoordTransformationGraph[RunType],
) -> ReducibleData[RunType]:
    """
    Computes coordinates, masks and corrections that are
    the same for the sample measurement and the reference measurement.
    """
    da = add_coords(da, graph)
    da = add_masks(da, ylim, zlims, bdlim, wbins)

    da = correct_by_footprint(da)

    # For some older Amor files there are no entries in the proton charge log
    if len(proton_charge) != 0:
        da = correct_by_proton_charge(da, proton_charge=proton_charge)

    return ReducibleData[RunType](da)


def default_parameters() -> dict:
    """Return the default parameters for the 2025 Amor workflow."""
    return {
        supermirror.MValue: sc.scalar(5, unit=sc.units.dimensionless),
        supermirror.CriticalEdge: 0.022 * sc.Unit("1/angstrom"),
        supermirror.Alpha: sc.scalar(0.25 / 0.088, unit=sc.units.angstrom),
        BeamSize[RunType]: 2.0 * sc.units.mm,
        DetectorSpatialResolution: 0.0025 * sc.units.m,
        Position[snx.NXsample, RunType]: sc.vector([0, 0, 0], unit="m"),
        NeXusDetectorName: "detector",
        ChopperPhase[RunType]: sc.scalar(7.0, unit="deg"),
        ChopperFrequency[RunType]: sc.scalar(8.333, unit="Hz"),
        BeamDivergenceLimits: (
            sc.scalar(-0.75, unit='deg'),
            sc.scalar(0.75, unit='deg'),
        ),
        GravityToggle: True,
        SampleRotationOffset[RunType]: sc.scalar(0.0, unit='deg'),
    }


providers = (
    load_detector,
    load_events,
    load_amor_ch_frequency,
    load_amor_ch_phase,
    load_amor_sample_rotation,
    load_amor_detector_rotation,
    load_amor_proton_charge,
    coordinate_transformation_graph,
    add_coords_masks_and_apply_corrections,
)
"""Providers that are specific to the file layout used until 2025."""
