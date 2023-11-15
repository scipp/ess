# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from typing import NewType, TypeVar, Generic
from dataclasses import dataclass

Up = NewType('Up', int)
Down = NewType('Down', int)
Unpolarized = NewType('Unpolarized', int)

Spin = TypeVar('Spin', bound=int)
PolarizerSpin = TypeVar('PolarizerSpin', bound=int)
AnalyzerSpin = TypeVar('AnalyzerSpin', bound=int)
CellType = TypeVar('CellType', bound=str)
Analyzer = NewType('Analyzer', str)
Polarizer = NewType('Polarizer', str)


@dataclass
class He3Transmission(Generic[CellType]):
    value: sc.DataArray


@dataclass
class He3CellPressure(Generic[CellType]):
    value: sc.Variable


@dataclass
class He3CellLength(Generic[CellType]):
    value: sc.Variable


@dataclass
class He3FillingTime(Generic[CellType]):
    value: sc.Variable


@dataclass
class He3Opacity(Generic[CellType]):
    value: sc.Variable


WavelengthBins = NewType('WavelengthBins', sc.Variable)
RawEventData = NewType('RawEventData', sc.DataArray)


def dummy_event_data() -> RawEventData:
    pass


@dataclass
class DirectBeamTimeIntervals(Generic[CellType, Spin]):
    value: sc.Variable


def dummy_time_intervals() -> DirectBeamTimeIntervals[CellType, Spin]:
    pass


@dataclass
class He3DirectBeam(Generic[CellType, Spin]):
    value: sc.DataArray


def he3_direct_beam(
    event_data: RawEventData,
    time_intervals: DirectBeamTimeIntervals[CellType, Spin],
    wavelength: WavelengthBins,
) -> He3DirectBeam[CellType, Spin]:
    """
    Returns the direct beam data for a given cell and spin state.

    We have a sequence of direct beam measurements, e.g., defined by a list of wall
    clock time intervals.
    """
    # compute wavelength
    # return event data, dims=(interval,wavelength)


def he3_opacity(
    pressure: He3CellPressure[CellType],
    cell_length: He3CellLength[CellType],
    wavelength: WavelengthBins,
) -> He3Opacity[CellType]:
    return He3Opacity[CellType]()


@dataclass
class He3InitialAtomicPolarization(Generic[CellType]):
    value: sc.DataArray


def he3_initial_atomic_polarization(
    direct_beam: He3DirectBeam[CellType, Unpolarized],
    opacity: He3Opacity[CellType],
    filling_time: He3FillingTime[CellType],
) -> He3InitialAtomicPolarization[CellType]:
    """
    Returns the initial atomic polarization for a given cell.

    The initial atomic polarization is computed from the direct beam data.
    """
    # results dims: spin state, wavelength
    return He3InitialAtomicPolarization[CellType](1)


def he3_transmission(
    opacity: He3Opacity[CellType],
    filling_time: He3FillingTime[CellType],
    direct_beam_up: He3DirectBeam[CellType, Up],
    direct_beam_down: He3DirectBeam[CellType, Down],
    initial_polarization: He3InitialAtomicPolarization[CellType],
) -> He3Transmission[CellType]:
    time_up = direct_beam_up.bins.coords['time'].mean()
    time_down = direct_beam_down.bins.coords['time'].mean()
    # results dims: spin state, wavelength, time
    return He3Transmission[CellType](1)


@dataclass
class SampleData(Generic[PolarizerSpin, AnalyzerSpin]):
    value: sc.DataArray


@dataclass
class PolarizationCorrectedSampleData(Generic[PolarizerSpin, AnalyzerSpin]):
    value: sc.DataArray


@dataclass
class SampleTimeIntervals(Generic[PolarizerSpin, AnalyzerSpin]):
    value: sc.Variable


def sample_data(
    event_data: RawEventData,
    time_intervals: SampleTimeIntervals[PolarizerSpin, AnalyzerSpin],
) -> SampleData[PolarizerSpin, AnalyzerSpin]:
    """
    Wavelength-dependent sample data for a given spin state.
    """
    pass


def polarization_corrected_sample_data(
    sample_data_up_up: SampleData[Up, Up],
    sample_data_up_down: SampleData[Up, Down],
    sample_data_down_up: SampleData[Down, Up],
    sample_data_down_down: SampleData[Down, Down],
    transmission_polarizer: He3Transmission[Polarizer],
    transmission_analyzer: He3Transmission[Analyzer],
) -> PolarizationCorrectedSampleData[PolarizerSpin, AnalyzerSpin]:
    # apply matrix inverse
    pass


def dummy_sample_time_intervals() -> SampleTimeIntervals[PolarizerSpin, AnalyzerSpin]:
    pass


providers = [
    dummy_event_data,
    dummy_sample_time_intervals,
    dummy_time_intervals,
    he3_direct_beam,
    he3_initial_atomic_polarization,
    he3_opacity,
    he3_transmission,
    polarization_corrected_sample_data,
    sample_data,
]
