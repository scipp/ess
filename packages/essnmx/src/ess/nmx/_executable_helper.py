# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import argparse
import enum
import glob
import logging
import pathlib
import sys
from functools import partial
from types import UnionType
from typing import Literal, TypeGuard, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from .types import Compression


def _validate_annotation(annotation) -> TypeGuard[type]:
    def _validate_atomic_type(annotation) -> bool:
        return (
            (annotation in (int, float, str, bool))
            or (isinstance(annotation, type) and issubclass(annotation, enum.StrEnum))
            or (get_origin(annotation) is Literal)
        )

    return (
        _validate_atomic_type(annotation)
        or (
            (origin := get_origin(annotation)) in (Union, UnionType)
            and _validate_atomic_type(_get_no_nonetype_args(annotation))
        )
        or (
            origin in (list, tuple, set)
            and len(args := get_args(annotation)) > 0
            and _validate_atomic_type(args[0])
        )
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
        field_info.annotation, enum.StrEnum
    ):
        return field_info.annotation[getattr(args, field_name)]
    return getattr(args, field_name)


def add_args_from_pydantic_model(
    *, model_cls: type[BaseModel], parser: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """Add arguments to the parser from the pydantic model class.

    Each field in the model class is added as a command line argument
    with the name `--{field-name}`.
    Arguments are added based on fields' information:
      - type annotation (type, choices, nargs)
      - description (help text)
      - default value (default, required and help text)

    Supported annotation for command arguments:
      - Atomic types: int, float, str, bool, enum.StrEnum, Literal
      - Optional[AtomicType]
      - List[AtomicType], Tuple[AtomicType, ...], Set[AtomicType]

    Parameters
    ----------
    model_cls:
        Pydantic model class to extract the arguments from.
    parser:
        Argument parser to add the arguments to.
        It adds a new argument group for the model.
        The group name is taken from the model's title config if available,
        otherwise the model class name is used.

    """
    group = parser.add_argument_group(
        model_cls.model_config.get("title", model_cls.__name__)
    )
    for field_name, field_info in model_cls.model_fields.items():
        add_argument = partial(group.add_argument, f"--{field_name.replace('_', '-')}")

        if not _validate_annotation(field_info.annotation):
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
        elif isinstance(arg_type, type) and issubclass(arg_type, enum.StrEnum):
            add_argument = partial(
                add_argument,
                type=str,
                choices=[str(e) for e in arg_type],
            )
            default = default.name if isinstance(default, enum.StrEnum) else default
        elif get_origin(arg_type) is Literal:
            add_argument = partial(
                add_argument,
                type=str,
                choices=[str(lit) for lit in get_args(arg_type)],
            )
        else:
            add_argument = partial(add_argument, type=arg_type, nargs=nargs)

        help_text = ' '.join([field_info.description or '', f"(default: {default})"])
        add_argument(default=default, required=required, help=help_text)

    return parser


class InputConfig(BaseModel):
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


class TimeBinUnit(enum.StrEnum):
    ms = 'ms'
    us = 'us'
    ns = 'ns'


class TimeBinCoordinate(enum.StrEnum):
    event_time_offset = 'event_time_offset'
    time_of_flight = 'time_of_flight'


class WorkflowConfig(BaseModel):
    # Add title of the basemodel
    model_config = {"title": "Workflow Configuration"}
    time_bin_coordinate: TimeBinCoordinate = Field(
        title="Time Bin Coordinate",
        description="Coordinate to bin the time data.",
        default=TimeBinCoordinate.event_time_offset,
    )
    nbins: int = Field(
        title="Number of Time Bins",
        description="Number of Time bins",
        default=50,
    )
    min_time_bin: int | None = Field(
        title="Minimum Time Bin",
        description="Minimum time edge of [time_bin_coordinate] in [time_bin_unit].",
        default=None,
    )
    max_time_bin: int | None = Field(
        title="Maximum Time Bin",
        description="Maximum time edge of [time_bin_coordinate] in [time_bin_unit].",
        default=None,
    )
    time_bin_unit: TimeBinUnit = Field(
        title="Unit of Time Bins",
        description="Unit of time bins.",
        default=TimeBinUnit.ms,
    )
    tof_lookup_table_file_path: str | None = Field(
        title="TOF Lookup Table File Path",
        description="Path to the TOF lookup table file. "
        "If None, the lookup table will be computed on-the-fly.",
        default=None,
    )
    tof_simulation_min_wavelength: float = Field(
        title="TOF Simulation Minimum Wavelength",
        description="Minimum wavelength for TOF simulation in Angstrom.",
        default=1.8,
    )
    tof_simulation_max_wavelength: float = Field(
        title="TOF Simulation Maximum Wavelength",
        description="Maximum wavelength for TOF simulation in Angstrom.",
        default=3.6,
    )
    tof_simulation_seed: int = Field(
        title="TOF Simulation Seed",
        description="Random seed for TOF simulation.",
        default=42,  # No reason.
    )


class OutputConfig(BaseModel):
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
    """Container for all reduction configurations."""

    inputs: InputConfig
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @property
    def _children(self) -> list[BaseModel]:
        return [self.inputs, self.workflow, self.output]


T = TypeVar('T', bound=BaseModel)


def from_args(cls: type[T], args: argparse.Namespace) -> T:
    """Create an instance of the pydantic model from the argparse namespace.

    It ignores any extra arguments in the namespace that are not part of the model.
    """
    kwargs = {
        field_name: _retrieve_field_value(field_name, field_info, args)
        for field_name, field_info in cls.model_fields.items()
    }
    return cls(**kwargs)


def build_reduction_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Command line arguments for the ESS NMX reduction. "
        "It assumes 14 Hz pulse speed."
    )
    parser = add_args_from_pydantic_model(model_cls=InputConfig, parser=parser)
    parser = add_args_from_pydantic_model(model_cls=WorkflowConfig, parser=parser)
    parser = add_args_from_pydantic_model(model_cls=OutputConfig, parser=parser)
    return parser


def reduction_config_from_args(args: argparse.Namespace) -> ReductionConfig:
    return ReductionConfig(
        inputs=from_args(InputConfig, args),
        workflow=from_args(WorkflowConfig, args),
        output=from_args(OutputConfig, args),
    )


def to_command_arguments(
    config: ReductionConfig, one_line: bool = True
) -> list[str] | str:
    """Convert the config to a list of command line arguments.

    Parameters
    ----------
    one_line:
        If True, return a single string with all arguments joined by spaces.
        If False, return a list of argument strings.

    """
    args = {}
    for instance in config._children:
        args.update(instance.model_dump(mode='python'))
    args = {f"--{k.replace('_', '-')}": v for k, v in args.items() if v is not None}

    arg_list = []
    for k, v in args.items():
        if not isinstance(v, bool):
            arg_list.append(k)
            if isinstance(v, list):
                arg_list.extend(str(item) for item in v)
            elif isinstance(v, enum.StrEnum):
                arg_list.append(v.value)
            else:
                arg_list.append(str(v))
        elif v is True:
            arg_list.append(k)

    if one_line:
        return ' '.join(arg_list)
    else:
        return arg_list


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
