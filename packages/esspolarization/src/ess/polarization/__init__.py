# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

# flake8: noqa
import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib


from .base import (
    CellInBeamLog,
    CellSpinLog,
    RunSectionLog,
    SampleInBeamLog,
    WavelengthBins,
    determine_run_section,
    extract_analyzer_direct_beam_polarized,
    extract_direct_beam,
    extract_polarizer_direct_beam_polarized,
    extract_sample_data_down_down,
    extract_sample_data_down_up,
    extract_sample_data_up_down,
    extract_sample_data_up_up,
)
from .base import providers as base_providers
from .base import run_reduction_workflow
from .he3 import (
    Depolarized,
    DirectBeamBackgroundQRange,
    DirectBeamNoCell,
    DirectBeamQRange,
    He3CellLength,
    He3CellPressure,
    He3DirectBeam,
    He3FillingTime,
    He3Opacity0,
    He3OpacityFunction,
    He3PolarizationFunction,
    He3TransmissionEmptyGlass,
    He3TransmissionFunction,
    Polarized,
    direct_beam,
    direct_beam_with_cell,
    get_he3_transmission_from_fit_to_direct_beam,
    he3_opacity_from_cell_params,
    he3_opacity_function_from_beam_data,
    he3_opacity_function_from_cell_opacity,
)
from .he3 import providers as he3_providers
from .types import (
    Analyzer,
    Down,
    PolarizationCorrectedData,
    Polarizer,
    PolarizingElement,
    ReducedSampleDataBySpinChannel,
    Up,
)

providers = base_providers + he3_providers

__all__ = [
    "Analyzer",
    "PolarizingElement",
    "CellInBeamLog",
    "CellSpinLog",
    "correct_sample_data_for_polarization",
    "Depolarized",
    "determine_run_section",
    "direct_beam",
    "DirectBeamBackgroundQRange",
    "DirectBeamNoCell",
    "DirectBeamQRange",
    "direct_beam_with_cell",
    "Down",
    "extract_analyzer_direct_beam_polarized",
    "extract_direct_beam",
    "extract_polarizer_direct_beam_polarized",
    "extract_sample_data_down_down",
    "extract_sample_data_down_up",
    "extract_sample_data_up_down",
    "extract_sample_data_up_up",
    "get_he3_transmission_from_fit_to_direct_beam",
    "He3CellLength",
    "He3CellPressure",
    "He3DirectBeam",
    "He3FillingTime",
    "He3Opacity0",
    "he3_opacity_from_cell_params",
    "He3OpacityFunction",
    "he3_opacity_function_from_beam_data",
    "he3_opacity_function_from_cell_opacity",
    "He3PolarizationFunction",
    "He3TransmissionEmptyGlass",
    "He3TransmissionFunction",
    "PolarizationCorrectedData",
    "Polarized",
    "Polarizer",
    "providers",
    "run_reduction_workflow",
    "ReducedSampleDataBySpinChannel",
    "RunSectionLog",
    "SampleInBeamLog",
    "Up",
    "WavelengthBins",
]
