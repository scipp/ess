# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

# flake8: noqa
import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

from .base import (
    Analyzer,
    Cell,
    CellInBeamLog,
    CellSpinLog,
    Depolarized,
    DirectBeamBackgroundQRange,
    DirectBeamNoCell,
    DirectBeamQRange,
    Down,
    He3CellLength,
    He3CellPressure,
    He3DirectBeam,
    He3FillingTime,
    He3Opacity,
    He3Polarization,
    He3Transmission,
    He3TransmissionEmptyGlass,
    PolarizationCorrectedSampleData,
    Polarized,
    Polarizer,
    RunSectionLog,
    SampleInBeamLog,
    Spin,
    SpinChannel,
    Up,
    WavelengthBins,
    correct_sample_data_for_polarization,
    determine_run_section,
    direct_beam,
    he3_opacity_from_beam_data,
    he3_opacity_from_cell_params,
    he3_polarization,
    he3_transmission,
    providers,
    spin_channel,
)

__all__ = [
    "providers",
    "correct_sample_data_for_polarization",
    "determine_run_section",
    "direct_beam",
    "he3_opacity_from_beam_data",
    "he3_opacity_from_cell_params",
    "he3_polarization",
    "he3_transmission",
    "spin_channel",
    "Up",
    "Down",
    "Depolarized",
    "Polarized",
    "Spin",
    "Analyzer",
    "Polarizer",
    "Cell",
    "WavelengthBins",
    "DirectBeamQRange",
    "DirectBeamBackgroundQRange",
    "He3Polarization",
    "He3Transmission",
    "He3CellPressure",
    "He3CellLength",
    "He3FillingTime",
    "He3Opacity",
    "He3TransmissionEmptyGlass",
    "DirectBeamNoCell",
    "He3DirectBeam",
    "PolarizationCorrectedSampleData",
    "CellSpinLog",
    "RunSectionLog",
    "SpinChannel",
    "SampleInBeamLog",
    "CellInBeamLog",
]
