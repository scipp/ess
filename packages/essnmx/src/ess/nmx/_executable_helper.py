# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import argparse
import logging
import sys
from typing import Literal

from pydantic import BaseModel

from .types import Compression


class InputConfig(BaseModel):
    # File IO
    input_file: str
    swmr: bool = False
    # Detector selection
    detector_ids: list[int | str] = [0, 1, 2]
    # Chunking options
    iter_chunk: bool = False
    chunk_size_pulse: int = 1
    chunk_size_events: int = 0

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("Input Options")
        group.add_argument(
            "--input-file", type=str, help="Path to the input file", required=True
        )
        group.add_argument(
            "--swmr", action="store_true", help="Open the input file in SWMR mode"
        )
        group.add_argument(
            "--detector-ids",
            type=int,
            nargs="+",
            default=[0, 1, 2],
            help="Detector indices to process",
        )
        chunk_option_group = parser.add_argument_group("Chunking Options")
        chunk_option_group.add_argument(
            "--iter-chunk",
            action="store_true",
            help="Whether to process the input file in chunks "
            " based on the hdf5 dataset chunk size. "
            "It is ignored if hdf5 dataset is not chunked. "
            "If True, it overrides chunk-size-pulse and chunk-size-events options.",
        )
        chunk_option_group.add_argument(
            "--chunk-size-pulse",
            type=int,
            default=0,
            help="Number of pulses to process in each chunk. "
            "If 0 or negative, process all pulses at once.",
        )
        chunk_option_group.add_argument(
            "--chunk-size-events",
            type=int,
            default=0,
            help="Number of events to process in each chunk. "
            "If 0 or negative, process all events at once."
            "If both chunk-size-pulse and chunk-size-events are set, "
            "chunk-size-pulse is preferred.",
        )
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "InputConfig":
        return cls(
            input_file=args.input_file,
            swmr=args.swmr,
            detector_ids=args.detector_ids,
            chunk_size_pulse=args.chunk_size_pulse,
            chunk_size_events=args.chunk_size_events,
            iter_chunk=args.iter_chunk,
        )


class WorkflowConfig(BaseModel):
    nbins: int = 50
    min_toa: int = 0
    max_toa: int = int((1 / 14) * 1_000)
    fast_axis: Literal['x', 'y'] | None = None

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("Workflow Options")
        group.add_argument(
            "--nbins",
            type=int,
            default=50,
            help="Number of TOF bins",
        )
        group.add_argument(
            "--min-toa",
            type=int,
            default=0,
            help="Minimum time of arrival (TOA) in [ms].",
        )
        group.add_argument(
            "--max-toa",
            type=int,
            default=int((1 / 14) * 1_000),
            help="Maximum time of arrival (TOA) in [ms].",
        )
        group.add_argument(
            "--fast-axis",
            type=str,
            choices=['x', 'y', None],
            default=None,
            help="Specify the fast axis of the detector. "
            "If None, it will be determined "
            "automatically based on the pixel offsets.",
        )
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "WorkflowConfig":
        return cls(
            nbins=args.nbins,
            min_toa=args.min_toa,
            max_toa=args.max_toa,
            fast_axis=args.fast_axis,
        )


class OutputConfig(BaseModel):
    # Log verbosity
    verbose: bool = False
    # File output
    output_file: str = "scipp_output.h5"
    compression: Compression = Compression.BITSHUFFLE_LZ4

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("Output Options")
        group.add_argument(
            "--verbose", "-v", action="store_true", help="Increase output verbosity"
        )
        group.add_argument(
            "--output-file",
            type=str,
            default="scipp_output.h5",
            help="Path to the output file",
        )
        group.add_argument(
            "--compression",
            type=str,
            default=Compression.BITSHUFFLE_LZ4.name,
            choices=[compression_key.name for compression_key in Compression],
            help="Compress option of reduced output file. Default: BITSHUFFLE_LZ4",
        )
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "OutputConfig":
        return cls(
            verbose=args.verbose,
            output_file=args.output_file,
            compression=Compression[args.compression],
        )


class ReductionConfig(BaseModel):
    inputs: InputConfig
    workflow: WorkflowConfig
    output: OutputConfig

    @classmethod
    def build_argument_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Command line arguments for the ESS NMX reduction. "
            "It assumes 14 Hz pulse speed."
        )
        parser = InputConfig.add_args(parser)
        parser = WorkflowConfig.add_args(parser)
        parser = OutputConfig.add_args(parser)
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ReductionConfig":
        return cls(
            inputs=InputConfig.from_args(args),
            workflow=WorkflowConfig.from_args(args),
            output=OutputConfig.from_args(args),
        )

    @property
    def _children(self) -> list[BaseModel]:
        return [self.inputs, self.workflow, self.output]

    def to_command_arguments(self) -> list[str]:
        args = {}
        for instance in self._children:
            args.update(instance.model_dump(mode='python'))
        args = {f"--{k.replace('_', '-')}": v for k, v in args.items()}

        arg_list = []
        for k, v in args.items():
            if not isinstance(v, bool):
                arg_list.append(k)
                if isinstance(v, list):
                    arg_list.extend(str(item) for item in v)
                elif isinstance(v, Compression):
                    arg_list.append(v.name)
                else:
                    arg_list.append(str(v))
            elif v is True:
                arg_list.append(k)

        return arg_list


def build_reduction_arg_parser() -> argparse.ArgumentParser:
    import warnings

    warnings.warn(
        "build_reduction_arg_parser is deprecated and will be removed "
        "in the future release (>=26.11.0) "
        "Please use the config classes to handle command line arguments.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(
        description="Command line arguments for the NMX reduction. "
        "It assumes 14 Hz pulse speed."
    )
    input_arg_group = parser.add_argument_group("Input Options")
    input_arg_group.add_argument(
        "--input_file", type=str, help="Path to the input file", required=True
    )
    input_arg_group.add_argument(
        "--nbins",
        type=int,
        default=50,
        help="Number of TOF bins",
    )
    input_arg_group.add_argument(
        "--detector_ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Detector indices to process",
    )

    output_arg_group = parser.add_argument_group("Output Options")
    output_arg_group.add_argument(
        "--output_file",
        type=str,
        default="scipp_output.h5",
        help="Path to the output file",
    )
    output_arg_group.add_argument(
        "--compression",
        type=str,
        default=Compression.BITSHUFFLE_LZ4.name,
        choices=[compression_key.name for compression_key in Compression],
        help="Compress option of reduced output file. Default: BITSHUFFLE_LZ4",
    )
    output_arg_group.add_argument(
        "--verbose", "-v", action="store_true", help="Increase output verbosity"
    )

    return parser


def build_logger(args: argparse.Namespace | OutputConfig) -> logging.Logger:
    logger = logging.getLogger(__name__)
    if args.verbose:
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger
