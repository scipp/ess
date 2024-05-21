# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Mapping, NewType, TypeVar

import numpy as np
import sciline as sl
import scipp as sc

from .he3 import (
    Analyzer,
    Cell,
    He3TransmissionFunction,
    Polarized,
    Polarizer,
    ReducedDirectBeamData,
    ReducedDirectBeamDataNoCell,
)

spin_up = sc.scalar(1, dtype='int64', unit=None)
spin_down = sc.scalar(-1, dtype='int64', unit=None)

Up = NewType('Up', int)
Down = NewType('Down', int)
PolarizerSpin = TypeVar('PolarizerSpin', Up, Down)
AnalyzerSpin = TypeVar('AnalyzerSpin', Up, Down)

WavelengthBins = NewType('WavelengthBins', sc.Variable)


PolarizationCorrectedSampleData = NewType(
    'PolarizationCorrectedSampleData', sc.DataArray
)
"""Polarization-corrected sample data."""


SampleInBeamLog = NewType('SampleInBeamLog', sc.DataArray)
"""Whether the sample is in the beam as a time series."""


class CellInBeamLog(sl.Scope[Cell, sc.DataArray], sc.DataArray):
    """Whether a given cell is in the beam as a time series."""


class CellSpinLog(sl.Scope[Cell, sc.DataArray], sc.DataArray):
    """Spin state of a given cell, as a time series."""


RunSectionLog = NewType('RunSectionLog', sc.Dataset)
"""
Run section as a time series.

Derived from several time-series logs in the NeXus file, e.g.,
whether the sample and cells are in the beam or not.
"""


def determine_run_section(
    sample_in_beam: SampleInBeamLog,
    polarizer_in_beam: CellInBeamLog[Polarizer],
    analyzer_in_beam: CellInBeamLog[Analyzer],
    polarizer_spin: CellSpinLog[Polarizer],
    analyzer_spin: CellSpinLog[Analyzer],
) -> RunSectionLog:
    from scipp.scipy.interpolate import interp1d

    logs = {
        'sample_in_beam': sample_in_beam,
        'polarizer_in_beam': polarizer_in_beam,
        'analyzer_in_beam': analyzer_in_beam,
        'polarizer_spin': polarizer_spin,
        'analyzer_spin': analyzer_spin,
    }
    # TODO Change this to datetime64
    times = [
        log.coords['time'].to(unit='s', dtype='float64', copy=False)
        for log in logs.values()
    ]
    times = sc.concat(times, 'time')
    times = sc.array(dims=times.dims, unit=times.unit, values=np.unique(times.values))
    logs = {
        name: interp1d(log, 'time', kind='previous', fill_value='extrapolate')(times)
        for name, log in logs.items()
    }
    return RunSectionLog(sc.Dataset(logs))


ReducedDataByRunSectionAndWavelength = NewType(
    'ReducedDataByRunSectionAndWavelength', sc.DataArray
)


def dummy_reduction(
    time_bands: sc.Variable,
    wavelength_bands: sc.Variable,
) -> sc.DataArray:
    """This is a placeholder returning meaningless data with correct shape."""
    data = time_bands[:-1] * wavelength_bands[:-1]
    data = data / data.sum()
    return sc.DataArray(
        data, coords={'time': time_bands, 'wavelength': wavelength_bands}
    )


def run_reduction_workflow(
    run_section: RunSectionLog,
    wavelength_bands: WavelengthBins,
) -> ReducedDataByRunSectionAndWavelength:
    """
    Run the reduction workflow.

    Note that is it currently not clear if we will wrap the workflow in a function,
    or assemble a common workflow. The structural details may thus be subject to
    change.

    The reduction workflow must return normalized event data, binned into time and
    wavelength bins. The time bands define intervals of different meaning, such as
    sample runs, direct beam runs, and spin states.
    """
    # TODO
    # Subdivide sample section into smaller intervals, or return numerator/denominator
    # separately? The latter would complicate things when supporting different
    # kinds of workflows, performing different kinds of normalizations.
    # We need to be careful when subdividing and (1) exactly preserve existing bounds
    # and (2) introduce new bounds using some heuristics that yield approximately
    # equal time intervals (for the sample runs).
    data = dummy_reduction(
        time_bands=run_section.coords['time'],
        wavelength_bands=wavelength_bands,
    )
    for name, log in run_section.items():
        data.coords[name] = log.data
    return ReducedDataByRunSectionAndWavelength(data)


def extract_direct_beam(
    data: ReducedDataByRunSectionAndWavelength,
) -> ReducedDirectBeamDataNoCell:
    """Extract direct beam without any cells from direct beam data."""
    is_direct_beam = ~(
        data.coords['sample_in_beam']
        | data.coords['polarizer_in_beam']
        | data.coords['analyzer_in_beam']
    )
    # We select all bins that correspond to direct-beam run sections. This preserves
    # the separation into distinct direct beam runs, which is required later for
    # fitting a time-decay function.
    return ReducedDirectBeamDataNoCell(data[is_direct_beam])


def extract_polarizer_direct_beam_polarized(
    data: ReducedDataByRunSectionAndWavelength,
) -> ReducedDirectBeamData[Polarizer, Polarized]:
    """Extract run sections with polarized polarizer from direct beam data."""
    # TODO We need all "polarized" runs, can we assume that
    # ReducedDataByRunSectionAndWavelength does not contain any depolarized data?
    select = (
        data.coords['polarizer_in_beam']
        & ~data.coords['sample_in_beam']
        & ~data.coords['analyzer_in_beam']
    )
    return ReducedDirectBeamData[Polarizer, Polarized](data[select])


def extract_analyzer_direct_beam_polarized(
    data: ReducedDataByRunSectionAndWavelength,
) -> ReducedDirectBeamData[Analyzer, Polarized]:
    """Extract run sections with polarized analyzer from direct beam data."""
    # TODO We need all "polarized" runs, can we assume that
    # ReducedDataByRunSectionAndWavelength does not contain any depolarized data?
    select = (
        data.coords['analyzer_in_beam']
        & ~data.coords['sample_in_beam']
        & ~data.coords['polarizer_in_beam']
    )
    return ReducedDirectBeamData[Analyzer, Polarized](data[select])


class ReducedSampleDataBySpinChannel(
    sl.ScopeTwoParams[PolarizerSpin, AnalyzerSpin, sc.DataArray], sc.DataArray
):
    """Sample data for a given spin channel."""


def is_sample_channel(
    coords: Mapping[str, sc.Variable],
    polarizer_spin: sc.Variable,
    analyzer_spin: sc.Variable,
) -> sc.Variable:
    return (
        coords['sample_in_beam']
        & coords['polarizer_in_beam']
        & coords['analyzer_in_beam']
        & (coords['polarizer_spin'] == polarizer_spin)
        & (coords['analyzer_spin'] == analyzer_spin)
    )


def extract_sample_data_up_up(
    data: ReducedDataByRunSectionAndWavelength,
) -> ReducedSampleDataBySpinChannel[Up, Up]:
    """Extract sample data for spin channel up-up."""
    return ReducedSampleDataBySpinChannel[Up, Up](
        is_sample_channel(data, spin_up, spin_up)
    )


def extract_sample_data_up_down(
    data: ReducedDataByRunSectionAndWavelength,
) -> ReducedSampleDataBySpinChannel[Up, Down]:
    """Extract sample data for spin channel up-down."""
    return ReducedSampleDataBySpinChannel[Up, Down](
        is_sample_channel(data, spin_up, spin_down)
    )


def extract_sample_data_down_up(
    data: ReducedDataByRunSectionAndWavelength,
) -> ReducedSampleDataBySpinChannel[Down, Up]:
    """Extract sample data for spin channel down-up."""
    return ReducedSampleDataBySpinChannel[Down, Up](
        is_sample_channel(data, spin_down, spin_up)
    )


def extract_sample_data_down_down(
    data: ReducedDataByRunSectionAndWavelength,
) -> ReducedSampleDataBySpinChannel[Down, Down]:
    """Extract sample data for spin channel down-down."""
    return ReducedSampleDataBySpinChannel[Down, Down](
        is_sample_channel(data, spin_down, spin_down)
    )


def correct_sample_data_for_polarization(
    upup: ReducedSampleDataBySpinChannel[Up, Up],
    updown: ReducedSampleDataBySpinChannel[Up, Down],
    downup: ReducedSampleDataBySpinChannel[Down, Up],
    downdown: ReducedSampleDataBySpinChannel[Down, Down],
    transmission_polarizer: He3TransmissionFunction[Polarizer],
    transmission_analyzer: He3TransmissionFunction[Analyzer],
) -> PolarizationCorrectedSampleData:
    """
    Apply polarization correction for the case of He3 polarizers and analyzers.

    There will be a different version of this function for handling the supermirror
    case, since transmission is not time-dependent but spin-flippers need to be
    accounted for.
    """
    # 1. Apply polarization correction (matrix inverse)
    # 2. Compute weighted mean over time and wavelength, bin into Q-bins

    # Pseudo code:
    # result = [0,0,0,0]
    # for j in range(4):
    #     for i, channel in enumerate((upup, updown, downup, downdown)):
    #         da = PA_inv[j, i] * channel
    #         da *= weights  # weights from error bars or event counts?
    #         result[j] += da.bins.concat('time', 'wavelength').hist(Qx=100, Qy=100)
    raise NotImplementedError()


providers = (
    determine_run_section,
    run_reduction_workflow,
    extract_direct_beam,
    extract_polarizer_direct_beam_polarized,
    extract_analyzer_direct_beam_polarized,
    extract_sample_data_down_down,
    extract_sample_data_down_up,
    extract_sample_data_up_down,
    extract_sample_data_up_up,
    correct_sample_data_for_polarization,
)
