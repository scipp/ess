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
from .configurations import OutputConfig, ReductionConfig
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

_TOF_COORD_NAME = 'tof'
"""Name of the TOF coordinate used in DataArrays."""


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
    tof_das = sc.DataGroup()
    detector_metas = sc.DataGroup()
    for detector_name in detector_names:
        cur_wf = base_wf.copy()
        cur_wf[NeXusName[snx.NXdetector]] = detector_name
        results = cur_wf.compute((TofDetector[SampleRun], NMXDetectorMetadata))
        detector_metas[detector_name] = results[NMXDetectorMetadata]
        # Binning into 1 bin and getting final tof bin edges later.
        tof_das[detector_name] = results[TofDetector[SampleRun]]

    # Make tof bin edges covering all detectors
    # TODO: Allow user to specify tof binning parameters from config
    min_tof = min(da.bins.coords[_TOF_COORD_NAME].min() for da in tof_das.values())
    max_tof = max(da.bins.coords[_TOF_COORD_NAME].max() for da in tof_das.values())
    n_edges = config.workflow.nbins + 1
    tof_bin_edges = sc.linspace(
        dim=_TOF_COORD_NAME, start=min_tof, stop=max_tof, num=n_edges
    )

    monitor_metadata = NMXMonitorMetadata(
        tof_bin_coord=_TOF_COORD_NAME,
        # TODO: Use real monitor data
        # Currently NMX simulations or experiments do not have monitors
        monitor_histogram=sc.DataArray(
            coords={_TOF_COORD_NAME: tof_bin_edges},
            data=sc.ones_like(tof_bin_edges[:-1]),
        ),
    )

    # Histogram detector counts
    tof_histograms = sc.DataGroup()
    for detector_name, tof_da in tof_das.items():
        histogram = tof_da.hist(tof=tof_bin_edges)
        tof_histograms[detector_name] = histogram

    results = sc.DataGroup(
        histogram=tof_histograms,
        detector=detector_metas,
        sample=metadatas[NMXSampleMetadata],
        source=metadatas[NMXSourceMetadata],
        monitor=monitor_metadata,
        lookup_table=base_wf.compute(TimeOfFlightLookupTable),
    )
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
