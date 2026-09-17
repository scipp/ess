# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import argparse
import logging
import warnings
from collections.abc import Callable

import scipp as sc
import scippnexus as snx

from ess.nmx._executable_helper import (
    add_args_from_pydantic_model,
    build_logger,
    from_args,
)
from ess.nmx.nexus import _check_file
from ess.nmx.types import (
    NMXDetectorMetadata,
    NMXInstrument,
    NMXLauetof,
    NMXMonitorMetadata,
    NMXReducedDetector,
    NMXSampleMetadata,
    NMXSourceMetadata,
)

from ._idf_helper import read_mandi_geometry_xml
from .configurations import (
    AuxiliaryOutputConfig,
    InputConfig,
    OutputConfig,
    ReductionConfig,
    WorkflowConfig,
)


def build_reduction_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Command line arguments for the Mandi reduction. "
        "It assumes 60 Hz pulse speed."
    )
    parser = add_args_from_pydantic_model(model_cls=InputConfig, parser=parser)
    parser = add_args_from_pydantic_model(model_cls=WorkflowConfig, parser=parser)
    parser = add_args_from_pydantic_model(model_cls=OutputConfig, parser=parser)
    parser = add_args_from_pydantic_model(
        model_cls=AuxiliaryOutputConfig, parser=parser
    )
    return parser


def reduction_config_from_args(args: argparse.Namespace) -> ReductionConfig:
    return ReductionConfig(
        inputs=from_args(InputConfig, args),
        workflow=from_args(WorkflowConfig, args),
        output=from_args(OutputConfig, args),
        aux=from_args(AuxiliaryOutputConfig, args),
    )


def _normalize_vector(vec: sc.Variable) -> sc.Variable:
    return vec / sc.norm(vec)


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

    t_coord_name = "tof"
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
                edge=min_t, coord_name=t_coord_name, desc='bigger'
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
                edge=max_t, coord_name=t_coord_name, desc='smaller'
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


def _retrieve_display(
    logger: logging.Logger | None, display: Callable | None
) -> Callable:
    if display is not None:
        return display
    elif logger is not None:
        return logger.info
    else:
        return logging.getLogger(__name__).info


def reduction(
    *,
    config: ReductionConfig,
    logger: logging.Logger | None = None,
    display: Callable | None = None,
) -> NMXLauetof:
    from ess.nmx.executables import save_results

    if not config.output.skip_file_output:
        _check_file(config.output.output_file, config.output.overwrite)
        config.aux.check_output_dir()

    display = _retrieve_display(logger, display)

    warnings.filterwarnings("ignore", category=UserWarning)
    # Loading
    with snx.File(config.inputs.input_file) as file:
        detectors = dict(
            filter(
                lambda kv: kv[0] not in config.inputs.ignore_list,
                file['entry/instrument'][snx.NXdetector].items(),
            )
        )
        total_detectors = len(detectors)
        banks = {}
        for idet, (name, det) in enumerate(detectors.items()):
            da = det[()]['events'].bins.concat().value.copy()
            # Mandi files' event_time_offset is time-of-flight
            da.coords['tof'] = da.coords.pop('event_time_offset')
            banks[name] = da
            display(f"{idet + 1}/{total_detectors} detector bank {name=} loaded.")

    mandi_geo = read_mandi_geometry_xml(config.inputs.input_file)
    detector_dict = {det.name: det for det in mandi_geo.detectors}
    tof_bin_edges = _build_mandi_time_bin_edges(wf_config=config.workflow, das=banks)
    det_hists = {}
    source_position = mandi_geo.source.position
    sample_position = mandi_geo.sample.position
    monitor_metadata = NMXMonitorMetadata(
        tof_bin_coord='tof',
        # TODO: Use real monitor data
        data=sc.DataArray(
            coords={'tof': tof_bin_edges},
            data=sc.ones_like(tof_bin_edges),
        ),
    )

    sample_meta = NMXSampleMetadata(
        # TODO: retrieve crystal rotation from the file correctly.
        crystal_rotation=sc.vector([0.0, 0.0, 0.0], unit='deg'),
        name=mandi_geo.sample.name,
        position=sample_position,
    )

    for ibank, (name, bank) in enumerate(banks.items()):
        if name not in detector_dict:
            warnings.warn(f"Detector {name=} not found in the IDF.", stacklevel=2)
            continue

        det_geo = detector_dict[name]
        binned = bank.group(det_geo.pixel_ids)
        hist = binned.hist(tof=tof_bin_edges.to(unit=bank.coords['tof'].unit))
        hist.coords['tof'] = hist.coords['tof'].to(
            unit=config.workflow.result_time_bin_unit
        )
        pixel_positions = det_geo.pixel_positions
        origin = pixel_positions.mean()
        distance = sc.norm(origin - source_position.to(unit=origin.unit))
        hist.coords['position'] = pixel_positions
        # We save the first pixel position so that DIALS can read use it.
        # first_pixel_position should be retrieved before folding.
        first_pixel_number = hist.coords['event_id'].min()
        first_pixel_position = hist['event_id', first_pixel_number].coords['position']
        first_pixel_position_from_sample = first_pixel_position - sample_position

        hist = det_geo.fold(hist)
        detector_meta = NMXDetectorMetadata(
            detector_name=name,
            x_pixel_size=det_geo.step_x,
            y_pixel_size=det_geo.step_y,
            origin=origin,
            fast_axis=_normalize_vector(det_geo.fast_axis),
            fast_axis_dim=det_geo.fast_axis_name + '_pixel_offset',
            slow_axis=_normalize_vector(det_geo.slow_axis),
            slow_axis_dim=det_geo.slow_axis_name + '_pixel_offset',
            distance=distance,
            first_pixel_position=first_pixel_position_from_sample,
        )
        det_hists[name] = NMXReducedDetector(data=hist, metadata=detector_meta)
        display(f"{ibank + 1}/{total_detectors} reduced")
        display(hist)

    instrument = NMXInstrument(
        instrument_definition=mandi_geo.instrument_definition,
        detectors=sc.DataGroup(det_hists),
        name="MANDI",
        source=NMXSourceMetadata(position=source_position),
    )
    results = NMXLauetof(
        control=monitor_metadata,
        instrument=instrument,
        sample=sample_meta,
    )
    if not config.output.skip_file_output:
        save_results(
            results=results,
            output_config=config.output,
            aux_config=config.aux,
            display=display,
        )
    return results


def main() -> None:
    parser = build_reduction_argument_parser()
    config = reduction_config_from_args(parser.parse_args())
    logger = build_logger(config.output)

    reduction(config=config, logger=logger)
