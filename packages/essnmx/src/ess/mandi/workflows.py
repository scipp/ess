# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippnexus as snx

from ess.nmx.types import (
    NMXDetectorMetadata,
    # NMXInstrument,
    # NMXLauetof,
    NMXSampleMetadata,
    NMXSourceMetadata,
)
from ess.reduce.nexus.types import (
    EmptyDetector,
    # Filename,
    NeXusComponent,
    NeXusTransformation,
    Position,
    # RunType,
    SampleRun,
)

from ._idf_helper import read_mandi_geometry_xml
from .configurations import (
    # AuxiliaryOutputConfig,
    # InputConfig,
    # OutputConfig,
    ReductionConfig,
    WorkflowConfig,
)


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


def _build_mandi_time_bin_edges(
    *, wf_config: WorkflowConfig, das: dict[str, sc.DataArray]
) -> sc.Variable:
    """Build time bin edges.

    Mostly copied from ess.nmx.executables module.
    However, for Mandi, we can build the time bin edges
    before we group/bin the event data.
    Therefore the helper function is slightly different.
    """
    import numpy as np

    from ess.nmx.executables import _warn_bin_edge_out_of_range

    t_coord_name = "time_of_flight"
    da_min_t = min(da.coords[t_coord_name].nanmin() for da in das.values())
    da_max_t = max(da.coords[t_coord_name].nanmax() for da in das.values())

    # Use the user-set parameters if available
    # and validate them according to the data.
    # Lower Time Bin Edge
    if wf_config.min_time_bin is not None:
        min_t = sc.scalar(wf_config.min_time_bin, unit=wf_config.time_bin_unit)
        min_t = min_t.to(unit=da_min_t.unit, dtype=da_min_t.dtype)
        # If the user-set minimum time bin value
        # is bigger than all time-bin-coordinate values.
        if min_t > da_max_t:
            _warn_bin_edge_out_of_range(
                edge=min_t, coord_name=wf_config.time_bin_coordinate, desc='bigger'
            )
    else:
        min_t = da_min_t

    # Upper Time Bin Edge
    if wf_config.max_time_bin is not None:
        max_t = sc.scalar(wf_config.max_time_bin, unit=wf_config.time_bin_unit)
        max_t = max_t.to(unit=da_max_t.unit, dtype=da_max_t.dtype)
        # If the user-set maximum time bin value
        # is smaller than all time-bin-coordinate values.
        if max_t <= da_min_t:
            _warn_bin_edge_out_of_range(
                edge=max_t, coord_name=wf_config.time_bin_coordinate, desc='smaller'
            )
    else:
        max_t = da_max_t

    # Validate the results.
    if min_t >= max_t:
        raise ValueError(
            f"Minimum time bin edge, {min_t} "
            "is bigger than or equal to the "
            f"maximum time bin edge, {max_t}.\n"
            "Cannot build a time bin edges coordinate.\n"
            "Please check your configurations again."
        )

    # If either min/max were manually selected and bin width is set.
    if wf_config.nbins is None:
        if wf_config.time_bin_width is None:
            time_bin_width = sc.scalar(300, unit='us').to(unit=wf_config.time_bin_unit)
        else:
            time_bin_width = sc.scalar(
                wf_config.time_bin_width, unit=wf_config.time_bin_unit
            )
        # We do not return a scalar bin width since we histogram
        # detector panels individually
        # and all histograms should have the same bin edges.
        min_t = min_t.to(unit=wf_config.time_bin_unit)
        max_t = max_t.to(unit=wf_config.time_bin_unit)
        bin_edges = sc.arange(
            dim=t_coord_name, start=min_t, stop=max_t, step=time_bin_width
        )
        # If the last bin edge is smaller than `max_t`
        if bin_edges[t_coord_name, -1] <= max_t:
            # Need to append one more edge to cover the whole range.
            true_last_bin_edge = bin_edges[t_coord_name, -1] + time_bin_width
            bin_edges = sc.concat([bin_edges, true_last_bin_edge], dim=t_coord_name)

        return bin_edges.to(dtype=float)

    else:  # Number of bin edges are given but not the bin width.
        n_edges = wf_config.nbins + 1
        if min_t.unit != max_t.unit:
            min_t = min_t.to(unit=wf_config.time_bin_unit)
            max_t = max_t.to(unit=wf_config.time_bin_unit)

        # Avoid dropping the event that has the exact same
        # `event_time_offset`` or `tof` value as the upper bin edge.
        max_t.value = np.nextafter(max_t.value, np.inf)
        return sc.linspace(
            dim=t_coord_name, start=min_t, stop=max_t, num=n_edges, dtype=float
        )


def reduce_mandi(*, config: ReductionConfig, logger, display) -> sc.DataGroup:
    import warnings

    try:
        import tqdm
    except ImportError:

        def tqdm(generator):
            if hasattr(generator, "__len__"):
                total_count = len(generator)
            for i_item, next_item in enumerate(generator):
                display(f"{i_item + 1}/{total_count}", next_item)
                yield next_item

    if config.output.verbose:
        progress = tqdm
    else:

        def progress(generator):
            yield from generator

    warnings.filterwarnings("ignore", category=UserWarning)
    # Loading
    with snx.File(config.inputs.input_file) as file:
        detectors = dict(
            filter(
                lambda kv: kv[0] not in config.inputs.ignore_list,
                file['entry/instrument'][snx.NXdetector].items(),
            )
        )
        banks = {
            name: det[()]['events'].bins.concat().value.copy()
            for name, det in progress(detectors.items())
        }
        # Mandi files' event_time_offset is time-of-flight
        for bank in banks.values():
            bank.coords['time_of_flight'] = bank.coords.pop('event_time_offset')

    mandi_geo = read_mandi_geometry_xml(config.inputs.input_file)
    detector_dict = {det.name: det for det in mandi_geo.detectors}
    tof_bin_edges = _build_mandi_time_bin_edges(wf_config=config.workflow, das=banks)
    results = {}
    for name, bank in progress(banks.items()):
        if name not in detector_dict:
            warnings.warn(f"Detector {name=} not found in the IDF.", stacklevel=2)
            continue

        det_geo = detector_dict[name]
        binned = bank.group(det_geo.pixel_ids)
        hist = binned.hist(
            time_of_flight=tof_bin_edges.to(unit=bank.coords['time_of_flight'].unit)
        )
        hist.coords['positions'] = det_geo.pixel_positions
        hist = det_geo.fold(hist)
        results[name] = hist

    return results
