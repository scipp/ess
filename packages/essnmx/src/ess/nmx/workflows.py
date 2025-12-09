# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pathlib
from collections.abc import Callable, Iterable

import pandas as pd
import sciline
import scipp as sc
import scippnexus as snx
import tof

from ess.reduce.nexus.types import (
    EmptyDetector,
    Filename,
    NeXusComponent,
    NeXusName,
    NeXusTransformation,
    Position,
    SampleRun,
)
from ess.reduce.time_of_flight import (
    DetectorLtotal,
    GenericTofWorkflow,
    LtotalRange,
    NumberOfSimulatedNeutrons,
    SimulationResults,
    SimulationSeed,
    TofLookupTableWorkflow,
)
from ess.reduce.time_of_flight.types import (
    TimeOfFlightLookupTable,
    TimeOfFlightLookupTableFilename,
)
from ess.reduce.workflow import register_workflow

from .configurations import WorkflowConfig
from .types import (
    NMXDetectorMetadata,
    NMXSampleMetadata,
    NMXSourceMetadata,
    TofSimulationMaxWavelength,
    TofSimulationMinWavelength,
)

default_parameters = {
    TofSimulationMaxWavelength: sc.scalar(3.6, unit='angstrom'),
    TofSimulationMinWavelength: sc.scalar(1.8, unit='angstrom'),
}


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


def _simulate_fixed_wavelength_tof(
    wmin: TofSimulationMinWavelength,
    wmax: TofSimulationMaxWavelength,
    ltotal_range: LtotalRange,
    neutrons: NumberOfSimulatedNeutrons,
    seed: SimulationSeed,
) -> SimulationResults:
    """
    Simulate a pulse of neutrons propagating through a chopper cascade using the
    ``tof`` package (https://tof.readthedocs.io).

    Parameters
    ----------
    """
    source = tof.Source(
        facility="ess", neutrons=neutrons, pulses=2, seed=seed, wmax=wmax, wmin=wmin
    )
    nmx_det = tof.Detector(distance=max(ltotal_range), name="detector")
    model = tof.Model(source=source, choppers=[], detectors=[nmx_det])
    results = model.run()
    events = results["detector"].data.squeeze().flatten(to="event")
    return SimulationResults(
        time_of_arrival=events.coords["toa"],
        speed=events.coords["speed"],
        wavelength=events.coords["wavelength"],
        weight=events.data,
        distance=results["detector"].distance,
    )


def _ltotal_range(detector_ltotal: DetectorLtotal[SampleRun]) -> LtotalRange:
    margin = sc.scalar(0.5, unit='m').to(
        unit=detector_ltotal.unit
    )  # Hardcoded margin of 50 cm. It's because the detector width is ~50 cm.
    ltotal_min = sc.min(detector_ltotal) - margin
    ltotal_max = sc.max(detector_ltotal) + margin
    return LtotalRange((ltotal_min, ltotal_max))


def patch_workflow_lookup_table_steps(*, wf: sciline.Pipeline) -> sciline.Pipeline:
    patched_wf = wf.copy()

    # Use TofLookupTableWorkflow
    patched_wf = _merge_workflows(patched_wf, TofLookupTableWorkflow())
    patched_wf.insert(_simulate_fixed_wavelength_tof)
    patched_wf.insert(_ltotal_range)
    return patched_wf


def _merge_panels(*da: sc.DataArray) -> sc.DataArray:
    """Merge multiple DataArrays representing different panels into one."""
    merged = sc.concat(da, dim='panel')
    return merged


def select_detector_names(
    *,
    input_files: list[pathlib.Path] | None = None,
    detector_ids: Iterable[int] = (0, 1, 2),
):
    if input_files is not None:
        detector_names = []
        # Collect all detector names from input files
        for input_file in input_files:
            with snx.File(input_file) as nexus_file:
                detector_names.extend(
                    nexus_file['entry/instrument'][snx.NXdetector].keys()
                )
        detector_names = sorted(set(detector_names))
        return [detector_names[i_d] for i_d in detector_ids]
    else:
        return ['detector_panel_0', 'detector_panel_1', 'detector_panel_2']


def map_detector_names(
    *,
    wf: sciline.Pipeline,
    detector_names: Iterable[str],
    mapped_type: type,
    reduce_func: Callable = _merge_panels,
) -> sciline.Pipeline:
    """Map detector indices(`panel`) to detector names in the workflow."""
    detector_name_map = pd.DataFrame({NeXusName[snx.NXdetector]: detector_names})
    detector_name_map.rename_axis(index='panel', inplace=True)
    wf[mapped_type] = wf[mapped_type].map(detector_name_map).reduce(func=reduce_func)
    return wf


def assemble_sample_metadata(
    crystal_rotation: Position[snx.NXcrystal, SampleRun],
    sample_position: Position[snx.NXsample, SampleRun],
    sample_component: NeXusComponent[snx.NXsample, SampleRun],
) -> NMXSampleMetadata:
    """Assemble sample metadata for NMX reduction workflow."""
    return NMXSampleMetadata(
        sample_name=sample_component['name'],
        crystal_rotation=crystal_rotation,
        sample_position=sample_position,
    )


def assemble_source_metadata(
    source_position: Position[snx.NXsource, SampleRun],
) -> NMXSourceMetadata:
    """Assemble source metadata for NMX reduction workflow."""
    return NMXSourceMetadata(source_position=source_position)


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

    return NMXDetectorMetadata(
        detector_name=detector_component['nexus_component_name'],
        x_pixel_size=x_pixel_size,
        y_pixel_size=y_pixel_size,
        origin_position=origin,
        fast_axis=_normalize_vector(fast_axis_vector),
        slow_axis=_normalize_vector(slow_axis_vector),
        distance=distance,
    )


@register_workflow
def NMXWorkflow() -> sciline.Pipeline:
    generic_wf = GenericTofWorkflow(run_types=[SampleRun], monitor_types=[])

    generic_wf.insert(_retrieve_crystal_rotation)
    generic_wf.insert(assemble_sample_metadata)
    generic_wf.insert(assemble_source_metadata)
    generic_wf.insert(assemble_detector_metadata)
    for key, value in default_parameters.items():
        generic_wf[key] = value

    return generic_wf


def compute_lookup_table(
    *,
    base_wf: sciline.Pipeline,
    workflow_config: WorkflowConfig,
    detector_names: Iterable[str],
) -> sc.DataArray:
    wf = base_wf.copy()
    if workflow_config.tof_lookup_table_file_path is not None:
        wf[TimeOfFlightLookupTableFilename] = workflow_config.tof_lookup_table_file_path
    else:
        wf = patch_workflow_lookup_table_steps(wf=wf)
        wmax = sc.scalar(workflow_config.tof_simulation_max_wavelength, unit='angstrom')
        wmin = sc.scalar(workflow_config.tof_simulation_min_wavelength, unit='angstrom')
        wf[TofSimulationMaxWavelength] = wmax
        wf[TofSimulationMinWavelength] = wmin
        wf[SimulationSeed] = workflow_config.tof_simulation_seed
        wf = map_detector_names(
            wf=wf,
            detector_names=detector_names,
            mapped_type=DetectorLtotal[SampleRun],
            reduce_func=_merge_panels,
        )

    return wf.compute(TimeOfFlightLookupTable)


__all__ = ['NMXWorkflow']
