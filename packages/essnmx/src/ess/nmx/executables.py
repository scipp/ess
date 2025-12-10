# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import logging
import pathlib
from collections.abc import Callable

import scipp as sc
import scippnexus as snx

from ess.reduce.nexus.types import Filename, NeXusName, SampleRun
from ess.reduce.time_of_flight.types import TimeOfFlightLookupTable, TofDetector

from ._executable_helper import (
    build_logger,
    build_reduction_argument_parser,
    collect_matching_input_files,
    reduction_config_from_args,
)
from .configurations import ReductionConfig, WorkflowConfig
from .nexus import (
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


def _finalize_tof_bin_edges(
    *, tof_das: sc.DataGroup, config: WorkflowConfig
) -> sc.Variable:
    tof_bin_edges = sc.concat(
        tuple(tof_da.coords['tof'] for tof_da in tof_das.values()), dim='tof'
    )
    return sc.linspace(
        dim='tof',
        start=sc.min(tof_bin_edges),
        stop=sc.max(tof_bin_edges),
        num=config.nbins + 1,
    )


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
    base_wf[TimeOfFlightLookupTable] = base_wf.compute(TimeOfFlightLookupTable)

    metadatas = base_wf.compute((NMXSampleMetadata, NMXSourceMetadata))
    export_static_metadata_as_nxlauetof(
        sample_metadata=metadatas[NMXSampleMetadata],
        source_metadata=metadatas[NMXSourceMetadata],
        output_file=config.output.output_file,
    )
    tof_das = sc.DataGroup()
    detector_metas = sc.DataGroup()
    for detector_name in detector_names:
        cur_wf = base_wf.copy()
        cur_wf[NeXusName[snx.NXdetector]] = detector_name
        results = cur_wf.compute((TofDetector[SampleRun], NMXDetectorMetadata))
        detector_meta: NMXDetectorMetadata = results[NMXDetectorMetadata]
        export_detector_metadata_as_nxlauetof(
            detector_metadata=detector_meta, output_file=config.output.output_file
        )
        detector_metas[detector_name] = detector_meta
        # Binning into 1 bin and getting final tof bin edges later.
        tof_das[detector_name] = results[TofDetector[SampleRun]].bin(tof=1)

    tof_bin_edges = _finalize_tof_bin_edges(tof_das=tof_das, config=config.workflow)

    monitor_metadata = NMXMonitorMetadata(
        tof_bin_coord='tof',
        # TODO: Use real monitor data
        # Currently NMX simulations or experiments do not have monitors
        monitor_histogram=sc.DataArray(
            coords={'tof': tof_bin_edges},
            data=sc.ones_like(tof_bin_edges[:-1]),
        ),
    )
    export_monitor_metadata_as_nxlauetof(
        monitor_metadata=monitor_metadata, output_file=config.output.output_file
    )

    # Histogram detector counts
    tof_histograms = sc.DataGroup()
    for detector_name, tof_da in tof_das.items():
        det_meta: NMXDetectorMetadata = detector_metas[detector_name]
        histogram = tof_da.hist(tof=tof_bin_edges)
        tof_histograms[detector_name] = histogram
        export_reduced_data_as_nxlauetof(
            detector_name=det_meta.detector_name,
            da=histogram,
            output_file=config.output.output_file,
            compress_mode=config.output.compression,
        )

    return sc.DataGroup(
        metadata=detector_metas,
        histogram=tof_histograms,
        lookup_table=base_wf.compute(TimeOfFlightLookupTable),
    )


def main() -> None:
    parser = build_reduction_argument_parser()
    config = reduction_config_from_args(parser.parse_args())
    logger = build_logger(config.output)

    reduction(config=config, logger=logger)
