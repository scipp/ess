# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Mapping, NewType, TypeVar

import numpy as np
import sciline as sl
import scipp as sc

from .uncertainty import broadcast_with_upper_bound_variances

spin_up = sc.scalar(1, dtype='int64', unit=None)
spin_down = sc.scalar(-1, dtype='int64', unit=None)

Depolarized = NewType('Depolarized', int)
Polarized = NewType('Polarized', int)
"""Polarized either up or down, don't care."""
PolarizationState = TypeVar('PolarizationState', Polarized, Depolarized)

Up = NewType('Up', int)
Down = NewType('Down', int)
PolarizerSpin = TypeVar('PolarizerSpin', Up, Down)
AnalyzerSpin = TypeVar('AnalyzerSpin', Up, Down)

Analyzer = NewType('Analyzer', str)
Polarizer = NewType('Polarizer', str)
Cell = TypeVar('Cell', Analyzer, Polarizer)

WavelengthBins = NewType('WavelengthBins', sc.Variable)

DirectBeamQRange = NewType('DirectBeamQRange', sc.Variable)
"""Q-range defining the direct beam region in a direct beam measurement."""

DirectBeamBackgroundQRange = NewType('DirectBeamBackgroundQRange', sc.Variable)
"""Q-range defining the direct beam background region in a direct beam measurement."""


class He3Polarization(sl.Scope[Cell, sc.DataArray], sc.DataArray):
    """Time-dependent polarization for a given cell."""


class He3Transmission(sl.Scope[Cell, sc.DataArray], sc.DataArray):
    """Wavelength- and time-dependent transmission for a given cell."""


class He3CellPressure(sl.Scope[Cell, sc.Variable], sc.Variable):
    """Pressure for a given cell."""


class He3CellLength(sl.Scope[Cell, sc.Variable], sc.Variable):
    """Length for a given cell."""


class He3CellTemperature(sl.Scope[Cell, sc.Variable], sc.Variable):
    """Temperature for a given cell."""


class He3FillingTime(sl.Scope[Cell, sc.Variable], sc.Variable):
    """Filling wall-clock time for a given cell."""


class He3OpacityFunction(sl.Scope[Cell, sc.DataArray], sc.DataArray):
    """Wavelength-dependent opacity function for a given cell."""


class He3TransmissionEmptyGlass(sl.Scope[Cell, sc.DataArray], sc.DataArray):
    """Transmission of the empty glass for a given cell."""


DirectBeamNoCell = NewType('DirectBeamNoCell', sc.DataArray)
"""Direct beam without cells and sample as a function of wavelength."""


class He3DirectBeam(
    sl.ScopeTwoParams[Cell, PolarizationState, sc.DataArray], sc.DataArray
):
    """
    Direct beam data for a given cell and spin state as a function of wavelength.
    """


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


def compute_direct_beam(
    data: sc.DataArray,
    q_range: sc.Variable,
    background_q_range: sc.Variable,
) -> sc.DataArray:
    """
    Compute background-subtracted direct beam function.

    The input must be normalized data, not counts.
    """
    if data.bins.unit != '':
        raise ValueError(f'Input data must be normalized, got unit {data.unit}.')
    if q_range.max() > background_q_range.min():
        raise ValueError('Background range must be after direct beam range.')
    if q_range.min() < sc.scalar(0.0, unit='1/angstrom'):
        raise ValueError('Q-range must be positive.')
    q_range = q_range**2
    background_q_range = background_q_range**2
    start_db = q_range[0]
    stop_db = q_range[-1]
    start_bg = background_q_range[0]
    stop_bg = background_q_range[-1]
    # Simple approach for now: Assume we can treat this as rotation invariant
    qx = data.bins.coords['Qx']
    qy = data.bins.coords['Qy']
    data.bins.coords['Q_squared'] = qx**2 + qy**2
    # The input is binned in time and wavelength, we simply take the per-bin mean
    # without changes.
    beam_region = data.bins['Q_squared', start_db:stop_db].bins.mean()
    background = data.bins['Q_squared', start_bg:stop_bg].bins.mean()
    return beam_region - background


ReducedDirectBeamDataNoCell = NewType('ReducedDirectBeamDataNoCell', sc.DataArray)


class ReducedDirectBeamData(
    sl.ScopeTwoParams[Cell, PolarizationState, sc.DataArray], sc.DataArray
):
    """Direct beam data for a given cell, as a function of wavelength and time."""


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


def direct_beam(
    data: ReducedDirectBeamDataNoCell,
    q_range: DirectBeamQRange,
    background_q_range: DirectBeamBackgroundQRange,
) -> DirectBeamNoCell:
    """
    Returns the direct beam function without any cells.

    The result is background-subtracted and returned as function of wavelength.
    Other dimensions of the input are preserved. In particular, the time dimension,
    corresponding to different direct beam measurements, is preserved.
    """
    return DirectBeamNoCell(
        compute_direct_beam(
            data=data,
            q_range=q_range,
            background_q_range=background_q_range,
        )
    )


def direct_beam_with_cell(
    data: ReducedDirectBeamData[Cell, PolarizationState],
    q_range: DirectBeamQRange,
    background_q_range: DirectBeamBackgroundQRange,
) -> He3DirectBeam[Cell, PolarizationState]:
    """
    Returns the direct beam function for a given cell.

    The result is background-subtracted and returned as function of wavelength and
    wall-clock time. The time dependence is coarse, i.e., due to different time
    intervals at which the direct beam is measured.
    """
    return He3DirectBeam[Cell, PolarizationState](
        compute_direct_beam(
            data=data,
            q_range=q_range,
            background_q_range=background_q_range,
        )
    )


class OpacityFunction:
    """Wavelength-dependent opacity function."""

    def __init__(self, opacity0: sc.Variable):
        self._opacity0 = opacity0.to(unit='1/Angstrom')

    @property
    def opacity0(self) -> sc.Variable:
        return self._opacity0

    def __call__(self, wavelength: sc.Variable) -> sc.Variable:
        scale = broadcast_with_upper_bound_variances(
            self.opacity0, sizes=wavelength.sizes
        )
        return (scale * wavelength).to(unit='', copy=False)


def he3_opacity_from_cell_params(
    pressure: He3CellPressure[Cell],
    length: He3CellLength[Cell],
    temperature: He3CellTemperature[Cell],
) -> He3OpacityFunction[Cell]:
    """
    Opacity function for a given cell, based on pressure and cell length.

    Note that this can alternatively be defined via neutron beam data, see
    :py:func:`he3_opacity_from_beam_data`.
    """
    from scipp.constants import Boltzmann as k_B

    he3_neutron_absorption_cross_section_at_1_angstrom = 2966.0e-24 * sc.Unit(
        'cm^2/Angstrom'
    )
    # Try to convert to Kelvin, since we get a more cryptic error message later if we
    # do not, in case the user passes in a temperature in degree Celsius.
    temperature = temperature.to(unit='K')

    opacity0 = (
        he3_neutron_absorption_cross_section_at_1_angstrom
        / (k_B * temperature)
        * pressure
        * length
    )
    return He3OpacityFunction[Cell](OpacityFunction(opacity0))


def he3_opacity_from_beam_data(
    transmission_empty_glass: He3TransmissionEmptyGlass[Cell],
    direct_beam: DirectBeamNoCell,
    direct_beam_cell: He3DirectBeam[Cell, Depolarized],
) -> He3OpacityFunction[Cell]:
    """
    Opacity function for a given cell, based on direct beam data.

    Note that this can alternatively be defined via cell parameters, see
    :py:func:`he3_opacity_from_cell_params`.
    """

    def intensity(wavelength: sc.Variable, opacity0: sc.Variable) -> sc.Variable:
        opacity = OpacityFunction(opacity0)
        return transmission_empty_glass * sc.exp(-opacity(wavelength))

    popt, _ = sc.curve_fit(
        ['wavelength'],
        intensity,
        direct_beam_cell / direct_beam,
        # TODO We could use opacity0 from cell parameters as initial guess.
        p0={'opacity0': sc.scalar(1.0, unit='1/nm')},
    )
    return He3OpacityFunction[Cell](OpacityFunction(popt['opacity0'].data))


def he3_polarization(
    direct_beam_no_cell: DirectBeamNoCell,
    direct_beam_polarized: He3DirectBeam[Cell, Polarized],
    opacity: He3OpacityFunction[Cell],
    filling_time: He3FillingTime[Cell],
    transmission_empty_glass: He3TransmissionEmptyGlass[Cell],
) -> He3Polarization[Cell]:
    """
    Fit time- and wavelength-dependent equation and return the fit param P(t).

    DB_pol/DB = T_E * cosh(O(lambda)*P(t))*exp(-O(lambda))
    """
    # Each time bin corresponds to a direct beam measurement. Take the mean for each
    # but keep the time binning.
    # time_up = direct_beam_up.bins.coords['time'].bins.mean()
    # time_down = direct_beam_down.bins.coords['time'].bins.mean()
    # results dims: spin state, wavelength, time
    raise NotImplementedError()
    return He3Polarization[Cell](1)


def he3_transmission(
    opacity: He3OpacityFunction[Cell],
    polarization: He3Polarization[Cell],
    transmission_empty_glass: He3TransmissionEmptyGlass[Cell],
) -> He3Transmission[Cell]:
    """
    Transmission for a given cell.

    This is computed from the opacity and polarization.
    """
    raise NotImplementedError()


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
    transmission_polarizer: He3Transmission[Polarizer],
    transmission_analyzer: He3Transmission[Analyzer],
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


providers = [
    determine_run_section,
    run_reduction_workflow,
    direct_beam,
    direct_beam_with_cell,
    extract_direct_beam,
    extract_polarizer_direct_beam_polarized,
    extract_analyzer_direct_beam_polarized,
    extract_sample_data_down_down,
    extract_sample_data_down_up,
    extract_sample_data_up_down,
    extract_sample_data_up_up,
    he3_transmission,
    he3_opacity_from_beam_data,
    he3_polarization,
    correct_sample_data_for_polarization,
]
