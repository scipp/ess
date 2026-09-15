# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import enum
import pathlib

from pydantic import BaseModel, Field, model_validator

from ess.nmx.types import Compression

# from ess.nmx.configurations import to_command_arguments


class InputConfig(BaseModel):
    # Add title of the basemodel
    model_config = {"title": "Input Configuration"}
    # File IO
    input_file: str = Field(title="Input File", description="Path to the input file.")
    swmr: bool = Field(
        title="SWMR Mode",
        description="Open the input file in SWMR mode",
        default=False,
    )
    # Detector selection
    ignore_list: list[str] = Field(
        title="Detector names to be excluded.",
        description="Detector indices to process",
        default=["bank_error", "bank_unmapped"],
    )


class TimeBinUnit(enum.StrEnum):
    ms = 'ms'
    us = 'us'
    ns = 'ns'


class _NotSet: ...


_notset = _NotSet()


class WorkflowConfig(BaseModel):
    # Add title of the basemodel
    @model_validator(mode='after')
    def nbins_or_time_bin_width(self):
        if self.time_bin_width is not None and self.nbins is not None:
            raise ValueError(
                "Either `nbins` or `time_bin_width` should be set. "
                "They cannot be set at the same time. "
                "It is allowed not setting any of them. "
                "Then 300 [us] of `time_bin_width` will be used."
            )
        return self

    @model_validator(mode='after')
    def positive_time_bin_width(self):
        if self.time_bin_width is not None and self.time_bin_width <= 0:
            raise ValueError("`time_bin_width` should be a positive number.")
        return self

    @model_validator(mode='after')
    def positive_nbins(self):
        if self.nbins is not None and self.nbins <= 0:
            raise ValueError("`nbins` should be a positive integer.")
        return self

    model_config = {"title": "Workflow Configuration"}
    time_bin_width: int | None = Field(
        title="Time Bin Width",
        description="Width(Length) of each Time Bin in [time_bin_unit]. "
        "If none of `time_bin_width` or `nbins` is given, "
        "300 [us] of `time_bin_width` will be used.",
        default=None,
    )
    nbins: int | None = Field(
        title="Number of Time Bins",
        description="Number of Time bins. ",
        default=None,
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
        default=TimeBinUnit.us,
    )
    result_time_bin_unit: TimeBinUnit = Field(
        title="Output Time Bin Unit",
        description="Time bin unit of the histogram after reduction. "
        "If the input time bin is different from the result time bin unit, "
        "the unit will be converted to the result time bin "
        "before the result is returned.",
        default=TimeBinUnit.ns,
        # DIALS expects [ns] by default.
    )


class AuxiliaryOutputConfig(BaseModel):
    # Add title of the basemodel
    model_config = {"title": "Auxiliary Output Configuration"}
    output_dir: str = Field(
        title="Path to the Auxiliary Files Directory",
        description="Directory to save auxiliary files into. "
        "If not given, stem of the output file name will be used.",
        default="",
    )

    @property
    def tof_1d_png_filename(self) -> str:
        """Hard-coded png file name for tof 1D histgoram plot."""
        return "essnmx-reduce-tof-1d.png"

    def build_target_dir(self, output_file: str = "") -> pathlib.Path:
        if self.output_dir:
            return pathlib.Path(self.output_dir)
        elif output_file:
            output_file_path = pathlib.Path(output_file)
            return output_file_path.parent / output_file_path.stem
        else:
            return pathlib.Path("essnmx-reduce-aux")

    def check_output_dir(self, output_file: str = "") -> None:
        """Raises if the expected auxiliary output directory path is invalid.

        Raises
        ------
            - If the parent directory does not exist.
            - If the path already exists but is not a directory.

        """
        target_dir = self.build_target_dir(output_file)
        if not target_dir.parent.is_dir():
            raise NotADirectoryError(
                "Parent directory doesn't exist "
                f"for the output files: {target_dir.parent}. "
                "Please make sure the parent directory exists first."
            )
        if target_dir.exists() and not target_dir.is_dir():
            raise NotADirectoryError(
                f"Target Directory path exists but it is not a directory: {target_dir} "
                "Please choose another directory path."
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
        default="scipp_mandi_output.h5",
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
    aux: AuxiliaryOutputConfig = Field(default_factory=AuxiliaryOutputConfig)

    @property
    def _children(self) -> list[BaseModel]:
        return [self.inputs, self.workflow, self.output, self.aux]
