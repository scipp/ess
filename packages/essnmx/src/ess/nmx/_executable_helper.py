# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import argparse
import glob
import logging
import pathlib
import sys
from enum import Enum
from functools import partial
from types import UnionType
from typing import Literal, Self, TypeGuard, Union, get_args, get_origin

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from .types import Compression


def _validate_annotation(annotation) -> TypeGuard[type]:
    return not (
        isinstance(annotation, type)
        or isinstance((origin_type := get_origin(annotation)), type)
        or (origin_type is UnionType)
        or (origin_type is Union)  # typing.Optional is Union[X, NoneType]
    )


def _get_no_nonetype_args(annotation) -> type:
    origin_type = get_origin(annotation)
    if (origin_type is UnionType or origin_type is Union) and type(None) in (
        union_args := get_args(annotation)
    ):
        arg_types = set(union_args) - {type(None)}
        if len(arg_types) > 1:
            raise TypeError(
                "Optional type with single non-None type is not supported: "
                f"{annotation}"
            )
        return next(iter(arg_types))
    return annotation


def _is_appendable_type(annotation) -> bool:
    return get_origin(annotation) in (list, tuple, set)


def _retrieve_field_value(
    field_name: str, field_info: FieldInfo, args: argparse.Namespace
):
    if isinstance(field_info.annotation, type) and issubclass(
        field_info.annotation, Enum
    ):
        return field_info.annotation[getattr(args, field_name)]
    return getattr(args, field_name)


class CommandArgument(BaseModel):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(cls.model_config.get("title", cls.__name__))
        for field_name, field_info in cls.model_fields.items():
            add_argument = partial(
                group.add_argument, f"--{field_name.replace('_', '-')}"
            )

            if _validate_annotation(field_info.annotation):
                raise TypeError(f"Unsupported annotation type: {field_info.annotation}")

            arg_type = _get_no_nonetype_args(field_info.annotation)
            if _is_appendable_type(arg_type):
                nargs = '+'
                arg_type = get_args(field_info.annotation)[0]
            else:
                nargs = None
                arg_type = arg_type

            required = field_info.default is PydanticUndefined
            default = ... if required else field_info.default

            if arg_type is bool:
                add_argument = partial(add_argument, action='store_true')
            elif isinstance(arg_type, type) and issubclass(arg_type, Enum):
                add_argument = partial(
                    add_argument,
                    type=str,
                    choices=[e.name for e in arg_type],
                )
                default = default.name if isinstance(default, Enum) else default
            elif get_origin(arg_type) is Literal:
                add_argument = partial(
                    add_argument,
                    type=str,
                    choices=[str(lit) for lit in get_args(arg_type)],
                )
            else:
                add_argument = partial(add_argument, type=arg_type, nargs=nargs)

            help_text = ' '.join(
                [field_info.description or '', f"(default: {default})"]
            )
            add_argument(default=default, required=required, help=help_text)

        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Self:
        kwargs = {
            field_name: _retrieve_field_value(field_name, field_info, args)
            for field_name, field_info in cls.model_fields.items()
        }
        return cls(**kwargs)


class InputConfig(CommandArgument, BaseModel):
    # Add title of the basemodel
    model_config = {"title": "Input Configuration"}
    # File IO
    input_file: list[str] = Field(
        title="Input File",
        description="Path to the input file. If multiple file paths are given,"
        " the output(histogram) will be merged(summed) "
        "and will not save individual outputs per input file. ",
    )
    swmr: bool = Field(
        title="SWMR Mode",
        description="Open the input file in SWMR mode",
        default=False,
    )
    # Detector selection
    detector_ids: list[int] = Field(
        title="Detector IDs",
        description="Detector indices to process",
        default=[0, 1, 2],
    )
    # Chunking options
    iter_chunk: bool = Field(
        title="Iterate in Chunks",
        description="Whether to process the input file in chunks "
        " based on the hdf5 dataset chunk size. "
        "It is ignored if hdf5 dataset is not chunked. "
        "If True, it overrides chunk-size-pulse and chunk-size-events options.",
        default=False,
    )
    chunk_size_pulse: int = Field(
        title="Chunk Size Pulse",
        description="Number of pulses to process in each chunk. "
        "If 0 or negative, process all pulses at once.",
        default=0,
    )
    chunk_size_events: int = Field(
        title="Chunk Size Events",
        description="Number of events to process in each chunk. "
        "If 0 or negative, process all events at once."
        "If both chunk-size-pulse and chunk-size-events are set, "
        "chunk-size-pulse is preferred.",
        default=0,
    )


class TOAUnit(Enum):
    ms = 'ms'
    us = 'us'
    ns = 'ns'


class WorkflowConfig(CommandArgument, BaseModel):
    # Add title of the basemodel
    model_config = {"title": "Workflow Configuration"}
    nbins: int = Field(
        title="Number of TOF Bins",
        description="Number of TOF bins",
        default=50,
    )
    min_toa: int = Field(
        title="Minimum Time of Arrival",
        description="Minimum time of arrival (TOA) in [toa_unit].",
        default=0,
    )
    max_toa: int = Field(
        title="Maximum Time of Arrival",
        description="Maximum time of arrival (TOA) in [toa_unit].",
        default=int((1 / 14) * 1_000),
    )
    toa_unit: TOAUnit = Field(
        title="Unit of TOA",
        description="Unit of TOA.",
        default=TOAUnit.ms,
    )
    fast_axis: Literal['x', 'y'] | None = Field(
        title="Fast Axis",
        description="Specify the fast axis of the detector. "
        "If None, it will be determined "
        "automatically based on the pixel offsets.",
        default=None,
    )


class OutputConfig(CommandArgument, BaseModel):
    # Add title of the basemodel
    model_config = {"title": "Output Configuration"}
    # Log verbosity
    verbose: bool = Field(
        title="Verbose Logging",
        description="Increase output verbosity.",
        default=False,
    )
    # File output
    output_file: str = Field(
        title="Output File",
        description="Path to the output file.",
        default="scipp_output.h5",
    )
    compression: Compression = Field(
        title="Compression",
        description="Compress option of reduced output file.",
        default=Compression.BITSHUFFLE_LZ4,
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

    def to_command_arguments(self, one_line: bool = True) -> list[str] | str:
        """Convert the config to a list of command line arguments.

        Parameters
        ----------
        one_line:
            If True, return a single string with all arguments joined by spaces.
            If False, return a list of argument strings.

        """
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
                elif isinstance(v, Enum):
                    arg_list.append(v.name)
                else:
                    arg_list.append(str(v))
            elif v is True:
                arg_list.append(k)

        if one_line:
            return ' '.join(arg_list)
        else:
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


def collect_matching_input_files(*input_file_patterns: str) -> list[pathlib.Path]:
    """Helper to collect input files matching the given patterns."""

    input_files: list[str] = []
    for pattern in input_file_patterns:
        input_files.extend(glob.glob(pattern))

    # Remove duplicates and sort
    return sorted({pathlib.Path(f).resolve() for f in input_files})
