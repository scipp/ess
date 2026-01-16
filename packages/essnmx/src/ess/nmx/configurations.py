# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import enum

from pydantic import BaseModel, Field

from .types import Compression


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
        description="Coordinate to bin the time data. "
        "Selecting `event_time_offset` means "
        "reduction steps are skipped, "
        "i.e. calculating `time of flight(tof)` "
        "and simply saves histograms of the raw data.",
        default=TimeBinCoordinate.time_of_flight,
        # Default is time of flight since
        # DIALS should expect the time of flight.
    )
    nbins: int = Field(
        title="Number of Time Bins",
        description="Number of Time bins",
        default=50,
    )
    min_time_bin: int | None = Field(
        title="Minimum Time",
        description="Minimum time edge of [time_bin_coordinate] in [time_bin_unit].",
        default=None,
    )
    max_time_bin: int | None = Field(
        title="Maximum Time",
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
    tof_simulation_num_neutrons: int = Field(
        title="Number of Neutrons for TOF Simulation",
        description="Number of neutrons to simulate for TOF lookup table calculation.",
        default=1_000_000,
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
    tof_simulation_min_ltotal: float = Field(
        title="TOF Simulation Minimum Ltotal",
        description="Minimum total flight path for TOF simulation in meters.",
        default=150.0,
    )
    tof_simulation_max_ltotal: float = Field(
        title="TOF Simulation Maximum Ltotal",
        description="Maximum total flight path for TOF simulation in meters.",
        default=170.0,
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
    skip_file_output: bool = Field(
        title="Skip File Output",
        description="If True, the output file will not be written.",
        default=False,
    )
    output_file: str = Field(
        title="Output File",
        description="Path to the output file. "
        "It will be overwritten if ``overwrite`` is True.",
        default="scipp_output.h5",
    )
    overwrite: bool = Field(
        title="Overwrite Output File",
        description="If True, overwrite the output file if ``output_file`` exists.",
        default=False,
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


def to_command_arguments(
    *, config: ReductionConfig, one_line: bool = True, separator: str = '\\\n'
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
        # Default separator is backslash + newline for better readability
        # Users can directly copy-paste the output in a terminal or a script.
        return (
            (separator + '--')
            .join(" ".join(arg_list).split('--'))
            .removeprefix(separator)
        )
    else:
        return arg_list
