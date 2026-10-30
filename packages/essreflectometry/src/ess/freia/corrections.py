# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import sciline
import scipp as sc
from ess.reduce.uncertainty import UncertaintyBroadcastMode

from ..reflectometry import corrections as common_corrections
from ..reflectometry.corrections import RunNormalization
from ..reflectometry.types import (
    BeamSize,
    CoordTransformationGraph,
    CorrectedDetector,
    ReducibleData,
    ReferenceRun,
    RunType,
    Sample,
    SampleRun,
    SampleSize,
    WavelengthDetector,
)
from .maskings import add_roi_masks
from .types import DetectorRegionOfInterest, WavelengthMonitor


def add_sample_coords_and_masks(
    da: WavelengthDetector[SampleRun],
    graph: CoordTransformationGraph[SampleRun],
    roi: DetectorRegionOfInterest[SampleRun],
) -> CorrectedDetector[SampleRun]:
    """Compute specular sample coordinates and mask events outside the ROI."""
    da = da.transform_coords(
        ['scattering_angle', 'theta', 'Q'],
        graph=graph,
        rename_dims=False,
        keep_intermediate=False,
    )
    return CorrectedDetector[SampleRun](add_roi_masks(da, roi))


def add_reference_coords_and_masks(
    da: WavelengthDetector[ReferenceRun],
    graph: CoordTransformationGraph[ReferenceRun],
    roi: DetectorRegionOfInterest[ReferenceRun],
) -> CorrectedDetector[ReferenceRun]:
    """Compute direct-beam coordinates and mask events outside the ROI."""
    da = da.transform_coords(
        # Direct-beam normalization needs the direction and sample normal.
        ['scattering_angle', 'outgoing_direction', 'sample_surface_normal'],
        graph=graph,
        rename_dims=False,
        keep_intermediate=False,
    )
    return CorrectedDetector[ReferenceRun](add_roi_masks(da, roi))


def add_offspecular_coords_and_masks(
    da: WavelengthDetector[SampleRun],
    graph: CoordTransformationGraph[SampleRun],
    roi: DetectorRegionOfInterest[SampleRun],
) -> CorrectedDetector[SampleRun]:
    """Compute off-specular coordinates and mask ROI and unassigned-beam events."""
    da = da.transform_coords(
        ['scattering_angle', 'incident_angle', 'reflection_angle', 'Qx', 'Qz'],
        graph=graph,
        rename_dims=False,
        keep_intermediate=False,
    )
    da = add_roi_masks(da, roi)
    # Closest-beam assignment with one candidate has scalar incidence.
    if 'incident_angle' in da.bins.coords:
        da = da.bins.assign_masks(
            incident_beam_unassigned=~sc.isfinite(da.bins.coords['incident_angle'])
        )
    return CorrectedDetector[SampleRun](da)


def normalize_by_monitor_histogram(
    detector: CorrectedDetector[RunType],
    *,
    monitor: WavelengthMonitor[RunType],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> ReducibleData[RunType]:
    """Normalize detector data by a histogrammed monitor.

    See :func:`ess.reduce.normalization.normalize_by_monitor_histogram` for the
    normalization and uncertainty treatment.
    """
    return common_corrections.normalize_by_monitor_histogram(
        detector=detector,
        monitor=monitor,
        uncertainty_broadcast_mode=uncertainty_broadcast_mode,
    )


def normalize_by_monitor_integrated(
    detector: CorrectedDetector[RunType],
    *,
    monitor: WavelengthMonitor[RunType],
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> ReducibleData[RunType]:
    """Normalize detector data by an integrated wavelength monitor."""
    return common_corrections.normalize_by_monitor_integrated(
        detector=detector,
        monitor=monitor,
        uncertainty_broadcast_mode=uncertainty_broadcast_mode,
    )


def insert_run_normalization(
    workflow: sciline.Pipeline, run_norm: RunNormalization
) -> None:
    """Insert providers for a specific FREIA run normalization into a workflow."""
    common_corrections.insert_run_normalization(
        workflow,
        run_norm,
        monitor_histogram_provider=normalize_by_monitor_histogram,
        monitor_integrated_provider=normalize_by_monitor_integrated,
    )


def correct_sample(
    sample: ReducibleData[SampleRun],
    beam_size: BeamSize[SampleRun],
    sample_size: SampleSize[SampleRun],
) -> Sample:
    """Apply the Gaussian footprint correction using the specular incidence angle.

    Beam size is the FWHM at the sample; sample size is its length along the beam.
    The correction applies only to the reflected run.
    """
    for name, size in [('BeamSize', beam_size), ('SampleSize', sample_size)]:
        if (
            not sc.isfinite(size).value
            or not (size > sc.scalar(0.0, unit=size.unit)).value
        ):
            raise ValueError(f'{name} must be finite and positive.')
    fraction = common_corrections.footprint_on_sample(
        sample.bins.coords['theta'], beam_size=beam_size, sample_size=sample_size
    )
    corrected = sample / fraction
    invalid = ~sc.isfinite(fraction) | (fraction <= sc.scalar(0.0))
    return Sample(corrected.bins.assign_masks(footprint=invalid))


providers = (
    add_sample_coords_and_masks,
    add_reference_coords_and_masks,
    correct_sample,
)
