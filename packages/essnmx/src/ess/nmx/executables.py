# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import logging
import pathlib
import warnings
from collections.abc import Callable

import numpy as np
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import Filename, NeXusName, RawDetector, SampleRun
from ess.reduce.time_of_flight.types import TimeOfFlightLookupTable, TofDetector

from ._executable_helper import (
    build_logger,
    build_reduction_argument_parser,
    collect_matching_input_files,
    reduction_config_from_args,
)
from .configurations import (
    OutputConfig,
    ReductionConfig,
    TimeBinCoordinate,
    WorkflowConfig,
)
from .nexus import (
    _check_file,
    export_detector_metadata_as_nxlauetof,
    export_monitor_metadata_as_nxlauetof,
    export_reduced_data_as_nxlauetof,
    export_static_metadata_as_nxlauetof,
)
from .types import (
    NMXDetectorMetadata,
    NMXMonitorMetadata,
    NMXSampleMetadata,
    NMXSourceMetadata,
)
from .workflows import initialize_nmx_workflow, select_detector_names

_TOF_COORD_NAME = 'tof'
"""Name of the TOF coordinate used in DataArrays."""
_ETO_COORD_NAME = 'event_time_offset'
"""Name of the Event Time Offset Coordinate used in Nexus."""


def _retrieve_input_file(input_file: list[str]) -> pathlib.Path:
    """Temporary helper to retrieve a single input file from the list
    Until multiple input file support is implemented.
    """
    if isinstance(input_file, list):
        input_files = collect_matching_input_files(*input_file)
        if len(input_files) == 0:
            raise ValueError(
                "No input files found for reduction."
                "Check if the file paths are correct.",
                input_file,
            )
        elif len(input_files) > 1:
            raise NotImplementedError(
                "Currently, only a single input file is supported for reduction."
            )
        input_file_path = input_files[0]
    else:
        input_file_path = input_file

    return input_file_path


def _retrieve_display(
    logger: logging.Logger | None, display: Callable | None
) -> Callable:
    if display is not None:
        return display
    elif logger is not None:
        return logger.info
    else:
        return logging.getLogger(__name__).info


def _retrieve_time_bin_coordinate_name(wf_config: WorkflowConfig) -> str:
    if wf_config.time_bin_coordinate == TimeBinCoordinate.time_of_flight:
        return _TOF_COORD_NAME
    elif wf_config.time_bin_coordinate == TimeBinCoordinate.event_time_offset:
        return _ETO_COORD_NAME


def _warn_bin_edge_out_of_range(
    *, edge: sc.Variable, coord_name: str, desc: str
) -> None:
    warnings.warn(
        message=f"{edge} is {desc} than all "
        f"{coord_name} values.\n"
        "The histogram will all have zero values.",
        category=UserWarning,
        stacklevel=4,
    )


def _match_data_unit_dtype(config_var: sc.Variable, da: sc.Variable) -> sc.Variable:
    return config_var.to(unit=da.unit, dtype=da.dtype)


def _build_time_bin_edges(
    *,
    wf_config: WorkflowConfig,
    result_das: sc.DataGroup,
    t_coord_name: str,
) -> sc.Variable:
    # Calculate the min and max of the data itself.
    da_min_t = min(da.bins.coords[t_coord_name].nanmin() for da in result_das.values())
    da_max_t = max(da.bins.coords[t_coord_name].nanmax() for da in result_das.values())

    # Use the user-set parameters if available
    # and validate them according to the data.
    # Lower Time Bin Edge
    if wf_config.min_time_bin is not None:
        min_t = sc.scalar(wf_config.min_time_bin, unit=wf_config.time_bin_unit)
        min_t = _match_data_unit_dtype(min_t, da=da_min_t)
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
        max_t = _match_data_unit_dtype(max_t, da=da_max_t)
        # If the user-set maximum time bin value
        # is smaller than all time-bin-coordinate values.
        if max_t <= da_min_t:
            _warn_bin_edge_out_of_range(
                edge=max_t, coord_name=wf_config.time_bin_coordinate, desc='smaller'
            )
    else:
        max_t = da_max_t

    # Avoid dropping the event that has the exact same
    # `event_time_offset`` or `tof` value as the upper bin edge.
    max_t.value = np.nextafter(max_t.value, np.inf)

    # Validate the results.
    if min_t >= max_t:
        raise ValueError(
            f"Minimum time bin edge, {min_t} "
            "is bigger than or equal to the "
            f"maximum time bin edge, {max_t}.\n"
            "Cannot build a time bin edges coordinate.\n"
            "Please check your configurations again."
        )

    # Build the bin-edges to histogram the results.
    n_edges = wf_config.nbins + 1
    return sc.linspace(dim=t_coord_name, start=min_t, stop=max_t, num=n_edges)


def reduction(
    *,
    config: ReductionConfig,
    logger: logging.Logger | None = None,
    display: Callable | None = None,
) -> sc.DataGroup:
    """Reduce NMX data from a Nexus file and export to NXLauetof(ESS NMX specific) file.

    Parameters
    ----------
    config:
        Reduction configuration.

        Data reduction parameters are taken from this config
        instead of passing them directly as keyword arguments.
        They can be either built from command-line arguments
        using `ReductionConfig.from_args()` or constructed manually.

        If the reduced data is successfully written to the output file
        the configuration is also saved there for future reference.
    logger:
        Logger to use for logging messages. If None, a default logger is created.
    display:
        Callable for displaying messages, useful in Jupyter notebooks. If None,
        defaults to logger.info.

    Returns
    -------
    sc.DataGroup:
        A DataGroup containing the reduced data for each selected detector.

    """
    # Check the file output configuration before we start heavy computation.
    _check_file(config.output.output_file, config.output.overwrite)

    display = _retrieve_display(logger, display)
    input_file_path = _retrieve_input_file(config.inputs.input_file).resolve()
    display(f"Input file: {input_file_path}")

    output_file_path = pathlib.Path(config.output.output_file).resolve()
    display(f"Output file: {output_file_path}")

    detector_names = select_detector_names(detector_ids=config.inputs.detector_ids)

    # Initialize workflow
    base_wf = initialize_nmx_workflow(config=config.workflow)
    # Insert parameters and cache intermediate results
    base_wf[Filename[SampleRun]] = input_file_path

    if config.workflow.time_bin_coordinate == TimeBinCoordinate.time_of_flight:
        # We cache the time of flight look up table
        # only if we need to calculate time-of-flight coordinates.
        # If `event_time_offset` was requested,
        # we do not have to calculate the look up table at all.
        base_wf[TimeOfFlightLookupTable] = base_wf.compute(TimeOfFlightLookupTable)

    metadatas = base_wf.compute((NMXSampleMetadata, NMXSourceMetadata))

    tof_das = sc.DataGroup()
    detector_metas = sc.DataGroup()

    if config.workflow.time_bin_coordinate == TimeBinCoordinate.event_time_offset:
        target_type = RawDetector[SampleRun]
    elif config.workflow.time_bin_coordinate == TimeBinCoordinate.time_of_flight:
        target_type = TofDetector[SampleRun]

    for detector_name in detector_names:
        cur_wf = base_wf.copy()
        cur_wf[NeXusName[snx.NXdetector]] = detector_name
        results = cur_wf.compute((target_type, NMXDetectorMetadata))
        detector_metas[detector_name] = results[NMXDetectorMetadata]
        # Binning into 1 bin and getting final tof bin edges later.
        tof_das[detector_name] = results[target_type]

    # Make tof bin edges covering all detectors
    t_coord_name = _retrieve_time_bin_coordinate_name(wf_config=config.workflow)
    t_bin_edges = _build_time_bin_edges(
        wf_config=config.workflow, result_das=tof_das, t_coord_name=t_coord_name
    )

    monitor_metadata = NMXMonitorMetadata(
        tof_bin_coord=t_coord_name,
        # TODO: Use real monitor data
        # Currently NMX simulations or experiments do not have monitors
        monitor_histogram=sc.DataArray(
            coords={t_coord_name: t_bin_edges},
            data=sc.ones_like(t_bin_edges[:-1]),
        ),
    )

    # Histogram detector counts
    tof_histograms = sc.DataGroup()
    for detector_name, tof_da in tof_das.items():
        histogram = tof_da.hist({t_coord_name: t_bin_edges})
        tof_histograms[detector_name] = histogram

    results = sc.DataGroup(
        histogram=tof_histograms,
        detector=detector_metas,
        sample=metadatas[NMXSampleMetadata],
        source=metadatas[NMXSourceMetadata],
        monitor=monitor_metadata,
    )

    if config.workflow.time_bin_coordinate == TimeBinCoordinate.time_of_flight:
        results["lookup_table"] = base_wf.compute(TimeOfFlightLookupTable)

    if not config.output.skip_file_output:
        save_results(results=results, output_config=config.output)

    return results


def save_results(*, results: sc.DataGroup, output_config: OutputConfig) -> None:
    # Validate if results have expected fields
    for mandatory_key in ['histogram', 'detector', 'sample', 'source', 'monitor']:
        if mandatory_key not in results:
            raise ValueError(f"Missing '{mandatory_key}' in results to save.")

    export_static_metadata_as_nxlauetof(
        sample_metadata=results['sample'],
        source_metadata=results['source'],
        output_file=output_config.output_file,
        overwrite=output_config.overwrite,
    )
    export_monitor_metadata_as_nxlauetof(
        monitor_metadata=results['monitor'],
        output_file=output_config.output_file,
    )
    for detector_name, detector_meta in results['detector'].items():
        export_detector_metadata_as_nxlauetof(
            detector_metadata=detector_meta,
            output_file=output_config.output_file,
        )
        export_reduced_data_as_nxlauetof(
            detector_name=detector_name,
            da=results['histogram'][detector_name],
            output_file=output_config.output_file,
            compress_mode=output_config.compression,
        )


def main() -> None:
    parser = build_reduction_argument_parser()
    config = reduction_config_from_args(parser.parse_args())
    logger = build_logger(config.output)

    reduction(config=config, logger=logger)
