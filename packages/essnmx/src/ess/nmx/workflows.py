# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Iterable

import sciline
import scipp as sc
import scippnexus as snx
from scippneutron.conversion.tof import tof_from_wavelength

from ess.reduce.nexus.types import (
    AnyRun,
    DiskChoppers,
    EmptyDetector,
    Filename,
    NeXusComponent,
    NeXusTransformation,
    Position,
    RunType,
    SampleRun,
)
from ess.reduce.unwrap import (
    GenericUnwrapWorkflow,
    LookupTableFilename,
    LookupTableRelativeErrorThreshold,
    LookupTableWorkflow,
    LtotalRange,
    SourcePosition,
    SourcePulse,
    WavelengthDetector,
)
from ess.reduce.workflow import register_workflow

from .configurations import WorkflowConfig
from .types import (
    NMXDetectorMetadata,
    NMXSampleMetadata,
    NMXSourceMetadata,
    TofDetector,
    TofSimulationMaxWavelength,
    TofSimulationMinWavelength,
)

default_parameters = {
    TofSimulationMaxWavelength: sc.scalar(3.6, unit='angstrom'),
    TofSimulationMinWavelength: sc.scalar(1.8, unit='angstrom'),
    LookupTableRelativeErrorThreshold: {f'detector_panel_{i}': 0.1 for i in range(5)},
    # TODO: This should become DiskChoppers[RunType] once we add choppers
    DiskChoppers[AnyRun]: {},
}


def select_detector_names(*, detector_ids: Iterable[int] = (0, 1, 2)):
    import os

    # Users can override detector names via environment variable
    # It is a comma-separated list of detector names
    # e.g., NMX_DETECTOR_NAMES=detector_panel_0,detector_panel_1,detector_panel_2
    # The detector names are not expected to be changed from the default ones,
    # but this option is provided for minimum flexibility.
    DETECTOR_NAME_VAR = os.environ.get("NMX_DETECTOR_NAMES", None)
    if DETECTOR_NAME_VAR is not None:
        return tuple(
            name
            for i_name, name in enumerate(DETECTOR_NAME_VAR.split(','))
            if i_name in detector_ids
        )
    else:
        return tuple(f'detector_panel_{i}' for i in detector_ids)


def assemble_sample_metadata(
    crystal_rotation: Position[snx.NXcrystal, SampleRun],
    sample_position: Position[snx.NXsample, SampleRun],
    sample_component: NeXusComponent[snx.NXsample, SampleRun],
) -> NMXSampleMetadata:
    """Assemble sample metadata for NMX reduction workflow."""
    name = sample_component['name']
    if isinstance(name, sc.Variable) and name.dtype == str:
        sample_name = name.value
    elif isinstance(name, str):
        sample_name = name
    else:
        raise TypeError(f'Sample name {name}is in a wrong type: ', type(name))

    return NMXSampleMetadata(
        name=sample_name,
        crystal_rotation=crystal_rotation,
        position=sample_position,
    )


def assemble_source_metadata(
    source_position: Position[snx.NXsource, SampleRun],
) -> NMXSourceMetadata:
    """Assemble source metadata for NMX reduction workflow."""
    return NMXSourceMetadata(position=source_position)


def _decide_fast_axis(da: sc.DataArray) -> str:
    x_slice = da['x_pixel_offset', 0].coords['detector_number']
    y_slice = da['y_pixel_offset', 0].coords['detector_number']

    if (x_slice.max() < y_slice.max()).value:
        return 'y'
    elif (x_slice.max() > y_slice.max()).value:
        return 'x'
    else:
        raise ValueError(
            "Cannot decide fast axis based on pixel offsets. "
            "Please specify the fast axis explicitly."
        )


def _decide_step(offsets: sc.Variable) -> sc.Variable:
    """Decide the step size based on the offsets assuming at least 2 values."""
    sorted_offsets = sc.sort(offsets, key=offsets.dim, order='ascending')
    return sorted_offsets[1] - sorted_offsets[0]


def _normalize_vector(vec: sc.Variable) -> sc.Variable:
    return vec / sc.norm(vec)


def _retrieve_crystal_rotation(
    file_path: Filename[SampleRun],
) -> Position[snx.NXcrystal, SampleRun]:
    """Temporary provider to retrieve crystal rotation from Nexus file."""
    from ess.reduce.nexus._nexus_loader import load_from_path
    from ess.reduce.nexus.types import NeXusLocationSpec

    spec = NeXusLocationSpec(
        filename=file_path,
        component_name='sample/crystal_rotation',
    )
    try:
        rotation: snx.nxtransformations.Transform = load_from_path(location=spec)
    except KeyError:
        import warnings

        warnings.warn(
            "No crystal rotation found in the Nexus file under "
            f"'entry/{spec.component_name}'. Returning zero rotation.",
            RuntimeWarning,
            stacklevel=1,
        )
        zero_rotation = sc.vector([0, 0, 0], unit='deg')
        return Position[snx.NXcrystal, SampleRun](zero_rotation)
    else:
        # TODO: Make sure if retrieving rotation vector is enough here.
        return Position[snx.NXcrystal, SampleRun](rotation.vector)


def assemble_detector_metadata(
    detector_component: NeXusComponent[snx.NXdetector, SampleRun],
    transformation: NeXusTransformation[snx.NXdetector, SampleRun],
    sample_position: Position[snx.NXsample, SampleRun],
    source_position: Position[snx.NXsource, SampleRun],
    empty_detector: EmptyDetector[SampleRun],
) -> NMXDetectorMetadata:
    """Assemble detector metadata for NMX reduction workflow."""
    positions = empty_detector.coords['position']
    # Origin should be the center of the detector.
    origin = positions.mean()
    _fast_axis = _decide_fast_axis(empty_detector)
    _slow_axis = 'y' if _fast_axis == 'x' else 'x'
    t_unit = transformation.value.unit

    axis_vectors = {
        'x': positions['x_pixel_offset', 1]['y_pixel_offset', 0]
        - positions['x_pixel_offset', 0]['y_pixel_offset', 0],
        'y': positions['y_pixel_offset', 1]['x_pixel_offset', 0]
        - positions['y_pixel_offset', 0]['x_pixel_offset', 0],
    }

    fast_axis_vector = axis_vectors[_fast_axis].to(unit=t_unit)
    slow_axis_vector = axis_vectors[_slow_axis].to(unit=t_unit)
    x_pixel_size = _decide_step(empty_detector.coords['x_pixel_offset'])
    y_pixel_size = _decide_step(empty_detector.coords['y_pixel_offset'])
    distance = sc.norm(origin - source_position.to(unit=origin.unit))

    # We save the first pixel position so that DIALS can read use it.
    flattened = empty_detector.flatten(to='detector_number')
    first_pixel_number = flattened.coords['detector_number'].min()
    first_pixel_position = flattened['detector_number', first_pixel_number].coords[
        'position'
    ]
    first_pixel_position_from_sample = first_pixel_position - sample_position

    return NMXDetectorMetadata(
        detector_name=detector_component['nexus_component_name'],
        x_pixel_size=x_pixel_size,
        y_pixel_size=y_pixel_size,
        origin=origin,
        fast_axis=_normalize_vector(fast_axis_vector),
        fast_axis_dim=_fast_axis + '_pixel_offset',
        slow_axis=_normalize_vector(slow_axis_vector),
        slow_axis_dim=_slow_axis + '_pixel_offset',
        distance=distance,
        first_pixel_position=first_pixel_position_from_sample,
    )


def compute_detector_tof(da: WavelengthDetector[RunType]) -> TofDetector[RunType]:
    """
    Compute the time-of-flight of neutrons from their wavelength.
    """
    return da.transform_coords(
        "tof", graph={"tof": tof_from_wavelength}, keep_intermediate=False
    )


def _source_position_to_SourcePosition(
    source_position: Position[snx.NXsource, SampleRun],
) -> SourcePosition:
    """
    This is a temporary provider to convert the source position from the Nexus file to
    the SourcePosition type used in the unwrapping workflow.
    In the next iteration, we will directly use the source position from the Nexus file
    in the unwrapping workflow and remove this provider.
    """
    return SourcePosition(source_position)


@register_workflow
def NMXWorkflow() -> sciline.Pipeline:
    generic_wf = GenericUnwrapWorkflow(run_types=[SampleRun], monitor_types=[])

    for provider in (
        _retrieve_crystal_rotation,
        assemble_sample_metadata,
        assemble_source_metadata,
        assemble_detector_metadata,
        compute_detector_tof,
        _source_position_to_SourcePosition,
    ):
        generic_wf.insert(provider)

    for key, value in default_parameters.items():
        generic_wf[key] = value

    return generic_wf


def _validate_mergable_workflow(wf: sciline.Pipeline):
    if wf.indices:
        raise NotImplementedError("Only flat workflow can be merged.")


def _merge_workflows(
    base_wf: sciline.Pipeline, merged_wf: sciline.Pipeline
) -> sciline.Pipeline:
    _validate_mergable_workflow(base_wf)
    _validate_mergable_workflow(merged_wf)

    for key, spec in merged_wf.underlying_graph.nodes.items():
        if 'value' in spec:
            base_wf[key] = spec['value']
        elif (provider_spec := spec.get('provider')) is not None:
            base_wf.insert(provider_spec.func)

    return base_wf


def initialize_nmx_workflow(*, config: WorkflowConfig) -> sciline.Pipeline:
    """Initialize NMX workflow according to the workflow configuration.

    If a lookup table file path is provided in the configuration,
    it is used directly. Otherwise, a TOF simulation workflow is added to
    the NMX workflow to compute the lookup table on-the-fly.

    All other parameters required for TOF simulation are also set
    as parameters in the workflow.

    Parameters
    ----------
    config:
        Workflow configuration for NMX reduction.
    params:
        Additional parameters to set in the workflow.

    """
    wf = NMXWorkflow()
    if config.lookup_table_file_path is not None:
        wf[LookupTableFilename] = config.lookup_table_file_path
    else:
        wf = _merge_workflows(base_wf=wf, merged_wf=LookupTableWorkflow())
        wmax = sc.scalar(config.tof_simulation_max_wavelength, unit='angstrom')
        wmin = sc.scalar(config.tof_simulation_min_wavelength, unit='angstrom')
        wf[SourcePulse] = SourcePulse(
            time=(sc.scalar(0.0, unit='ms'), sc.scalar(5.0, unit='ms')),
            wavelength=(wmin, wmax),
        )
        ltotal_min = sc.scalar(value=config.tof_simulation_min_ltotal, unit='m')
        ltotal_max = sc.scalar(value=config.tof_simulation_max_ltotal, unit='m')
        wf[LtotalRange] = LtotalRange((ltotal_min, ltotal_max))

    return wf


__all__ = ['NMXWorkflow']
