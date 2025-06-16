# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import argparse
import logging
import pathlib
import sys
from collections.abc import Callable
from functools import partial

import sciline as sl
import scipp as sc

from ess.reduce.streaming import (
    EternalAccumulator,
    MaxAccumulator,
    MinAccumulator,
    StreamProcessor,
)

from ..nexus import (
    _export_detector_metadata_as_nxlauetof,
    _export_reduced_data_as_nxlauetof,
    _export_static_metadata_as_nxlauetof,
)
from ..streaming import calculate_number_of_chunks
from ..types import (
    DetectorIndex,
    DetectorName,
    FilePath,
    MaximumCounts,
    MaximumProbability,
    MaximumTimeOfArrival,
    McStasWeight2CountScaleFactor,
    MinimumTimeOfArrival,
    NMXDetectorMetadata,
    NMXExperimentMetadata,
    NMXRawDataMetadata,
    NMXReducedCounts,
    NMXReducedDataGroup,
    PixelIds,
    RawEventProbability,
    TimeBinSteps,
)
from . import McStasWorkflow
from .load import (
    mcstas_weight_to_probability_scalefactor,
    raw_event_data_chunk_generator,
)
from .xml import McStasInstrument


def _build_metadata_streaming_processor_helper() -> (
    Callable[[sl.Pipeline], StreamProcessor]
):
    return partial(
        StreamProcessor,
        dynamic_keys=(RawEventProbability,),
        target_keys=(NMXRawDataMetadata,),
        accumulators={
            MaximumProbability: MaxAccumulator,
            MaximumTimeOfArrival: MaxAccumulator,
            MinimumTimeOfArrival: MinAccumulator,
        },
    )


def _build_final_streaming_processor_helper() -> (
    Callable[[sl.Pipeline], StreamProcessor]
):
    return partial(
        StreamProcessor,
        dynamic_keys=(RawEventProbability,),
        target_keys=(NMXReducedDataGroup,),
        accumulators={NMXReducedCounts: EternalAccumulator},
    )


def calculate_raw_data_metadata(
    *detector_ids: DetectorIndex | DetectorName,
    wf: sl.Pipeline,
    chunk_size: int = 10_000_000,
    logger: logging.Logger | None = None,
) -> NMXRawDataMetadata:
    # Stream processor building helper
    scalefactor_stream_processor = _build_metadata_streaming_processor_helper()
    metadata_wf = wf.copy()
    # Loop over the detectors
    file_path = metadata_wf.compute(FilePath)
    raw_data_metadatas = {}

    for detector_i in detector_ids:
        temp_wf = metadata_wf.copy()
        if isinstance(detector_i, str):
            temp_wf[DetectorName] = detector_i
        else:
            temp_wf[DetectorIndex] = detector_i

        detector_name = temp_wf.compute(DetectorName)
        max_chunk_id = calculate_number_of_chunks(
            temp_wf.compute(FilePath),
            detector_name=detector_name,
            chunk_size=chunk_size,
        )
        # Build the stream processor
        processor = scalefactor_stream_processor(temp_wf)
        for i_da, da in enumerate(
            raw_event_data_chunk_generator(
                file_path=file_path, detector_name=detector_name, chunk_size=chunk_size
            )
        ):
            if any(da.sizes.values()) == 0:
                continue
            else:
                results = processor.add_chunk({RawEventProbability: da})
            if logger is not None:
                logger.info(
                    "[{%s}/{%s}] Processed chunk for {%s}",
                    i_da + 1,
                    max_chunk_id,
                    detector_name,
                )

        raw_data_metadatas[detector_i] = results[NMXRawDataMetadata]

    # We take the min/maximum values of the scale factor
    # We are doing it manually because it is not possible to update parameters
    # in the workflow that stream processor uses.
    min_toa = min(dg.min_toa for dg in raw_data_metadatas.values())
    max_toa = max(dg.max_toa for dg in raw_data_metadatas.values())
    max_probability = max(dg.max_probability for dg in raw_data_metadatas.values())

    return NMXRawDataMetadata(
        min_toa=min_toa, max_toa=max_toa, max_probability=max_probability
    )


def reduction(
    *,
    input_file: pathlib.Path,
    output_file: pathlib.Path,
    chunk_size: int = 10_000_000,
    nbins: int = 51,
    max_counts: int | None = None,
    detector_ids: list[int | str],
    compression: bool = True,
    wf: sl.Pipeline | None = None,
    logger: logging.Logger | None = None,
    toa_min_max_prob: tuple[float] | None = None,
) -> None:
    wf = wf.copy() if wf is not None else McStasWorkflow()
    wf[FilePath] = input_file
    # Set static info
    wf[McStasInstrument] = wf.compute(McStasInstrument)

    if not toa_min_max_prob:
        # Calculate parameters for data reduction
        data_metadata = calculate_raw_data_metadata(
            *detector_ids, wf=wf, logger=logger, chunk_size=chunk_size
        )
        if logger is not None:
            logger.info("Metadata retrieved: %s", data_metadata)

        toa_bin_edges = sc.linspace(
            dim='t', start=data_metadata.min_toa, stop=data_metadata.max_toa, num=nbins
        )
        scale_factor = mcstas_weight_to_probability_scalefactor(
            max_counts=wf.compute(MaximumCounts),
            max_probability=data_metadata.max_probability,
        )
    else:
        if logger is not None:
            logger.info("Metadata given: %s", toa_min_max_prob)
        toa_min = sc.scalar(toa_min_max_prob[0], unit='s')
        toa_max = sc.scalar(toa_min_max_prob[1], unit='s')
        prob_max = sc.scalar(toa_min_max_prob[2])
        toa_bin_edges = sc.linspace(dim='t', start=toa_min, stop=toa_max, num=nbins)
        scale_factor = mcstas_weight_to_probability_scalefactor(
            max_counts=wf.compute(MaximumCounts),
            max_probability=prob_max,
        )

    if max_counts:
        scale_factor = mcstas_weight_to_probability_scalefactor(
            max_counts=MaximumCounts(max_counts),
            max_probability=data_metadata.max_probability,
        )
    else:
        scale_factor = mcstas_weight_to_probability_scalefactor(
            max_counts=wf.compute(MaximumCounts),
            max_probability=data_metadata.max_probability,
        )
    # Compute metadata and make the skeleton output file
    experiment_metadata = wf.compute(NMXExperimentMetadata)
    detector_metas = []
    for detector_i in range(3):
        temp_wf = wf.copy()
        temp_wf[DetectorIndex] = detector_i
        detector_metas.append(temp_wf.compute(NMXDetectorMetadata))

    if logger is not None:
        logger.info("Exporting metadata into the output file %s", output_file)

    _export_static_metadata_as_nxlauetof(
        experiment_metadata=experiment_metadata,
        output_file=output_file,
        # Arbitrary metadata falls into ``entry`` group as a variable.
        mcstas_weight2count_scale_factor=scale_factor,
    )
    _export_detector_metadata_as_nxlauetof(*detector_metas, output_file=output_file)
    # Compute histogram
    final_wf = wf.copy()
    # Set the scale factor and time bin edges
    final_wf[McStasWeight2CountScaleFactor] = scale_factor
    final_wf[TimeBinSteps] = toa_bin_edges

    file_path = final_wf.compute(FilePath)
    final_stream_processor = _build_final_streaming_processor_helper()
    # Loop over the detectors
    result_list = []
    for detector_i in detector_ids:
        temp_wf = final_wf.copy()
        if isinstance(detector_i, str):
            temp_wf[DetectorName] = detector_i
        else:
            temp_wf[DetectorIndex] = detector_i
        # Set static information as parameters
        detector_name = temp_wf.compute(DetectorName)
        temp_wf[PixelIds] = temp_wf.compute(PixelIds)
        max_chunk_id = calculate_number_of_chunks(
            file_path, detector_name=detector_name, chunk_size=chunk_size
        )

        # Build the stream processor
        processor = final_stream_processor(temp_wf)
        for i_da, da in enumerate(
            raw_event_data_chunk_generator(
                file_path=file_path, detector_name=detector_name, chunk_size=chunk_size
            )
        ):
            if any(da.sizes.values()) == 0:
                continue
            else:
                results = processor.add_chunk({RawEventProbability: da})
            if logger is not None:
                logger.info(
                    "[{%s}/{%s}] Processed chunk for {%s}",
                    i_da + 1,
                    max_chunk_id,
                    detector_name,
                )

        result = results[NMXReducedDataGroup]
        result_list.append(result)
        if logger is not None:
            logger.info("Appending reduced data into the output file %s", output_file)
        _export_reduced_data_as_nxlauetof(
            result, output_file=output_file, compress_counts=compression
        )
    from ess.nmx.reduction import merge_panels

    return merge_panels(*result_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="McStas Data Reduction.")
    parser.add_argument(
        "--input_file", type=str, help="Path to the input file", required=True
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="scipp_output.h5",
        help="Path to the output file",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Increase output verbosity"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10_000_000,
        help="Chunk size for processing",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=51,
        help="Number of TOF bins",
    )
    parser.add_argument(
        "--max_counts",
        type=int,
        default=None,
        help="Maximum Counts",
    )
    parser.add_argument(
        "--detector_ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Detector indices to process",
    )
    parser.add_argument(
        "--compression",
        type=bool,
        default=True,
        help="Compress reduced output with bitshuffle/lz4",
    )

    args = parser.parse_args()

    input_file = pathlib.Path(args.input_file).resolve()
    output_file = pathlib.Path(args.output_file).resolve()

    logger = logging.getLogger(__name__)
    if args.verbose:
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))

    wf = McStasWorkflow()
    reduction(
        input_file=input_file,
        output_file=output_file,
        chunk_size=args.chunk_size,
        nbins=args.nbins,
        max_counts=args.max_counts,
        detector_ids=args.detector_ids,
        compression=args.compression,
        logger=logger,
        wf=wf,
    )
