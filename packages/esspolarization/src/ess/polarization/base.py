# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, TypeVar

import sciline as sl
import scipp as sc

Up = NewType('Up', int)
Down = NewType('Down', int)
Unpolarized = NewType('Unpolarized', int)
Spin = TypeVar('Spin', Up, Down, Unpolarized)

Analyzer = NewType('Analyzer', str)
Polarizer = NewType('Polarizer', str)
Cell = TypeVar('Cell', Analyzer, Polarizer)

WavelengthBins = NewType('WavelengthBins', sc.Variable)
RawEventData = NewType('RawEventData', sc.DataArray)

DirectBeamRegion = NewType('DirectBeamRegion', sc.Variable)
"""ROI for the direct beam region in a direct beam measurement."""

DirectBeamBackgroundRegion = NewType('DirectBeamBackgroundRegion', sc.Variable)
"""ROI for the direct beam background region in a direct beam measurement."""


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


class He3TransmissionEmptyGlass(sl.Scope[Cell, sc.DataArray], sc.DataArray):
    """Transmission of the empty glass for a given cell."""


DirectBeam = NewType('DirectBeam', sc.DataArray)
"""Direct beam without cells and sample as a function of wavelength."""


class He3DirectBeam(sl.ScopeTwoParams[Cell, Spin, sc.DataArray], sc.DataArray):
    """
    Direct beam data for a given cell and spin state as a function of wavelength.
    """


class He3InitialAtomicPolarization(sl.Scope[Cell, sc.DataArray], sc.DataArray):
    """
    Initial atomic polarization for a given cell.

    This is computed from the direct beam data with an unpolarized cell.
    """


SampleData = NewType('SampleData', sc.DataArray)
"""
Uncorrected sample data.

TODO How exactly is this defined? Which normal (non-polarization) corrections
have been applied already?
"""


PolarizationCorrectedSampleData = NewType(
    'PolarizationCorrectedSampleData', sc.DataArray
)
"""Polarization-corrected sample data."""


class CellSpin(sl.Scope[Cell, sc.DataArray], sc.DataArray):
    """Spin state of a given cell, as a time series."""


RunSection = NewType('RunSection', sc.DataArray)
"""
Run type (sample/direct/neither/...) as a time series.

This needs to be derived from some time-series logs in the NeXus file, e.g.,
whether the sample is in the beam or not.
"""


SpinChannel = NewType('SpinChannel', sc.DataArray)
"""Time series of the combined spin channel (++, +-, -+, --)."""

RawDataByRunSection = NewType('RawDataByRunSection', sc.DataArray)
"""Raw event data with events labeled (or grouped) by run section (sample/direct)."""

DirectBeamData = NewType('DirectBeamData', sc.DataArray)
"""
Raw direct beam event data with events labeled (or grouped) by cell and spin state.
"""


def spin_channel(
    polarizer_spin: CellSpin[Polarizer], analyzer_spin: CellSpin[Analyzer]
) -> SpinChannel:
    """
    Returns a time series of the combined spin channel (++, +-, -+, --).

    This will be used to split the raw detector data into the four spin channels.
    """
    # TODO In practice, are switching times instant? Do we need to drop events that
    # occur during switching? How is this marked in the meta data?
    return SpinChannel()


def direct_beam(
    event_data: DirectBeamData,
    wavelength: WavelengthBins,
    direct_beam_region: DirectBeamRegion,
    direct_beam_background_region: DirectBeamBackgroundRegion,
) -> DirectBeam:
    """
    Extract direct beam without any cells from direct beam data.

    The result is background-subtracted and returned as function of wavelength.
    """
    return DirectBeam()


def he3_direct_beam(
    event_data: DirectBeamData,
    wavelength: WavelengthBins,
    direct_beam_region: DirectBeamRegion,
    direct_beam_background_region: DirectBeamBackgroundRegion,
) -> He3DirectBeam[Cell, Spin]:
    """
    Returns the direct beam data for a given cell and spin state.

    The result is background-subtracted and returned as function of wavelength.
    """
    return He3DirectBeam[Cell, Spin]()


def he3_opacity_from_cell_params(
    pressure: He3CellPressure[Cell],
    cell_length: He3CellLength[Cell],
    wavelength: WavelengthBins,
) -> He3Opacity[Cell]:
    """
    Opacity for a given cell, based on pressure and cell length.

    Note that this can alternatively be computed from neutron beam data, see
    :py:func:`he3_opacity_from_beam_data`.
    """
    # TODO What is this magic number?
    return He3Opacity[Cell](0.07733 * pressure * cell_length * wavelength)


def he3_opacity_from_beam_data(
    transmission_empty_glass: He3TransmissionEmptyGlass[Cell],
    direct_beam: DirectBeam,
    direct_beam_cell: He3DirectBeam[Cell, Unpolarized],
) -> He3Opacity[Cell]:
    """
    Opacity for a given cell, based on direct beam data.

    Note that this can alternatively be computed from cell parameters, see
    :py:func:`he3_opacity_from_cell_params`.
    """
    # TODO What is I_bg? Is it also computed from the direct beam data?
    return He3Opacity[Cell]()


def he3_initial_atomic_polarization(
    direct_beam: DirectBeam,
    direct_beam_up: He3DirectBeam[Cell, Up],
    direct_beam_down: He3DirectBeam[Cell, Down],
    transmission_empty_glass: He3TransmissionEmptyGlass[Cell],
    opacity: He3Opacity[Cell],
) -> He3InitialAtomicPolarization[Cell]:
    """
    Returns the initial atomic polarization for a given cell.

    The initial atomic polarization is computed from the direct beam data.

    Note that we could use either direct_beam_up or direct_beam_down here, since the
    initial atomic polarization is the same for both.

    Note also that if He3Opacity is computed from beam params certain terms would
    cancel here, but this is available only at the end of the cell lifetime. Therefore
    we compute this based on the opacity, which may be computed alternatively from
    cell params.
    """
    # results dims: spin state, wavelength
    return He3InitialAtomicPolarization[Cell](1)


def he3_transmission(
    opacity: He3Opacity[Cell],
    filling_time: He3FillingTime[Cell],
    initial_polarization: He3InitialAtomicPolarization[Cell],
    transmission_empty_glass: He3TransmissionEmptyGlass[Cell],
) -> He3Transmission[Cell]:
    # Each time bin corresponds to a direct beam measurement. Take the mean for each
    # but keep the time binning.
    # time_up = direct_beam_up.bins.coords['time'].bins.mean()
    # time_down = direct_beam_down.bins.coords['time'].bins.mean()
    # results dims: spin state, wavelength, time
    return He3Transmission[Cell](1)


def raw_data_by_run_section(
    raw: RawEventData, run_section: RunSection
) -> RawDataByRunSection:
    """
    Split raw event data into sample data and direct beam data.

    This assigns each raw event to either the sample run or the direct beam run,
    based in the time-dependent run type.

    The reason for splitting the data here before splitting into spin channels and the
    various direct runs is to avoid performance issues from multiple passes over the
    full data set.
    """
    return RawDataByRunSection()


def sample_data_by_spin_channel(
    event_data: RawDataByRunSection,
    spin_channel: SpinChannel,
) -> SampleData:
    """
    Wavelength-dependent sample data for all spin channels.

    This labels event with their spin channel and maybe groups by channel, so we have
    an output dimension of length 4 (++, +-, -+, --).
    """
    pass


def direct_beam_data_by_cell_and_polarization(
    event_data: RawDataByRunSection,
    polarizer_spin: CellSpin[Polarizer],
    analyzer_spin: CellSpin[Analyzer],
) -> DirectBeamData:
    """ """


def polarization_corrected_sample_data(
    sample_data: SampleData,
    transmission_polarizer: He3Transmission[Polarizer],
    transmission_analyzer: He3Transmission[Analyzer],
) -> PolarizationCorrectedSampleData:
    """
    Apply polarization correction for the case of He3 polarizers and analyzers.

    There will be a different version of this function for handling the supermirror
    case, since transmission is not time-dependent but spin-flippers need to be
    accounted for.
    """
    # apply matrix inverse
    pass


providers = [
    direct_beam,
    he3_direct_beam,
    he3_initial_atomic_polarization,
    he3_opacity_from_beam_data,
    he3_transmission,
    polarization_corrected_sample_data,
    sample_data_by_spin_channel,
    spin_channel,
    raw_data_by_run_section,
    direct_beam_data_by_cell_and_polarization,
]
