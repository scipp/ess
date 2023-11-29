# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, TypeVar

import sciline as sl
import scipp as sc

Up = NewType('Up', int)
Down = NewType('Down', int)
Unpolarized = NewType('Unpolarized', int)
Spin = TypeVar('Spin', Up, Down, Unpolarized)
PolarizerSpin = TypeVar('PolarizerSpin', Up, Down)
AnalyzerSpin = TypeVar('AnalyzerSpin', Up, Down)

Analyzer = NewType('Analyzer', str)
Polarizer = NewType('Polarizer', str)
Cell = TypeVar('Cell', Analyzer, Polarizer)

WavelengthBins = NewType('WavelengthBins', sc.Variable)
RawEventData = NewType('RawEventData', sc.DataArray)


class He3Transmission(sl.Scope[Cell, sc.DataArray], sc.DataArray):
    """Spin-, Time-, and wavelength-dependent transmission for a given cell."""


class He3CellPressure(sl.Scope[Cell, sc.Variable], sc.Variable):
    """Pressure for a given cell."""


class He3CellLength(sl.Scope[Cell, sc.Variable], sc.Variable):
    """Length for a given cell."""


class He3FillingTime(sl.Scope[Cell, sc.Variable], sc.Variable):
    """Filling wall-clock time for a given cell."""


class He3Opacity(sl.Scope[Cell, sc.DataArray], sc.DataArray):
    """Wavelength-dependent opacity for a given cell."""


class DirectBeamTimeIntervals(sl.ScopeTwoParams[Cell, Spin, sc.Variable], sc.Variable):
    """
    Wall-clock time intervals for a given cell.

    This defines the time intervals that correspond the direct beam measurements. This
    is used to extract the direct beam data from the total event data.
    """


class He3DirectBeam(sl.ScopeTwoParams[Cell, Spin, sc.DataArray], sc.DataArray):
    """
    Direct beam data for a given cell and spin state.

    TODO How is this defined? It is a processed version of the raw direct beam event
    data.
    """


class He3InitialAtomicPolarization(sl.Scope[Cell, sc.DataArray], sc.DataArray):
    """
    Initial atomic polarization for a given cell.

    This is computed from the direct beam data with an unpolarized cell.
    """


class SampleData(
    sl.ScopeTwoParams[PolarizerSpin, AnalyzerSpin, sc.DataArray], sc.DataArray
):
    """
    Uncorrected sample data.

    TODO How exactly is this defined? Which normal (non-polarization) corrections
    have been applied already?
    """


class PolarizationCorrectedSampleData(
    sl.ScopeTwoParams[PolarizerSpin, AnalyzerSpin, sc.DataArray], sc.DataArray
):
    """Polarization-corrected sample data."""


class SampleTimeIntervals(
    sl.ScopeTwoParams[PolarizerSpin, AnalyzerSpin, sc.Variable], sc.Variable
):
    """
    Wall-clock time intervals for a given sample.

    This defines the time intervals that correspond the sample measurements. This is
    used to extract the sample data from the total event data.
    """


def dummy_event_data() -> RawEventData:
    pass


def dummy_time_intervals() -> DirectBeamTimeIntervals[Cell, Spin]:
    pass


def he3_direct_beam(
    event_data: RawEventData,
    time_intervals: DirectBeamTimeIntervals[Cell, Spin],
    wavelength: WavelengthBins,
) -> He3DirectBeam[Cell, Spin]:
    """
    Returns the direct beam data for a given cell and spin state.

    We have a sequence of direct beam measurements, e.g., defined by a list of wall
    clock time intervals.
    """
    # compute wavelength
    # return event data, dims=(interval,wavelength)


def he3_opacity(
    pressure: He3CellPressure[Cell],
    cell_length: He3CellLength[Cell],
    wavelength: WavelengthBins,
) -> He3Opacity[Cell]:
    return He3Opacity[Cell]()


def he3_initial_atomic_polarization(
    direct_beam: He3DirectBeam[Cell, Unpolarized],
    opacity: He3Opacity[Cell],
    filling_time: He3FillingTime[Cell],
) -> He3InitialAtomicPolarization[Cell]:
    """
    Returns the initial atomic polarization for a given cell.

    The initial atomic polarization is computed from the direct beam data.
    """
    # results dims: spin state, wavelength
    return He3InitialAtomicPolarization[Cell](1)


def he3_transmission(
    opacity: He3Opacity[Cell],
    filling_time: He3FillingTime[Cell],
    direct_beam_up: He3DirectBeam[Cell, Up],
    direct_beam_down: He3DirectBeam[Cell, Down],
    initial_polarization: He3InitialAtomicPolarization[Cell],
) -> He3Transmission[Cell]:
    # Each time bin corresponds to a direct beam measurement. Take the mean for each
    # but keep the time binning.
    # time_up = direct_beam_up.bins.coords['time'].bins.mean()
    # time_down = direct_beam_down.bins.coords['time'].bins.mean()
    # results dims: spin state, wavelength, time
    return He3Transmission[Cell](1)


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
    """
    Apply polarization correction for the case of He3 polarizers and analyzers.

    There will be a different version of this function for handling the supermirror
    case, since transmission is not time-dependent but spin-flippers need to be
    accounted for.
    """
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
