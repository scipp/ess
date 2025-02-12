# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import Generic, NewType, TypeVar

import sciline as sl
import scipp as sc

from ess.reduce.uncertainty import broadcast_with_upper_bound_variances

from .types import (
    Analyzer,
    AnalyzerSpin,
    Down,
    PlusMinus,
    PolarizerSpin,
    PolarizingElement,
    TransmissionFunction,
    Up,
)

Depolarized = NewType('Depolarized', int)
Polarized = NewType('Polarized', int)
"""Polarized either up or down, don't care."""
PolarizationState = TypeVar('PolarizationState', Polarized, Depolarized)


DirectBeamNoCell = NewType('DirectBeamNoCell', sc.DataArray)
"""Direct beam without cells and sample as a function of wavelength."""


class He3CellPressure(sl.Scope[PolarizingElement, sc.Variable], sc.Variable):
    """Pressure for a given cell."""


class He3CellLength(sl.Scope[PolarizingElement, sc.Variable], sc.Variable):
    """Length for a given cell."""


class He3CellTemperature(sl.Scope[PolarizingElement, sc.Variable], sc.Variable):
    """Temperature for a given cell."""


class He3FillingTime(sl.Scope[PolarizingElement, sc.Variable], sc.Variable):
    """Filling wall-clock time for a given cell."""


class He3CellTransmissionFractionIncomingUnpolarized(
    sl.ScopeTwoParams[PolarizingElement, PolarizationState, sc.DataArray], sc.DataArray
):
    """Transmission fraction for a given cell"""


class He3AnalyzerTransmissionFractionIncomingPolarized(
    sl.ScopeTwoParams[PolarizerSpin, AnalyzerSpin, sc.DataArray], sc.DataArray
):
    """Transmission fraction of the analyzer with polarized incoming beam"""


He3AnalyzerTransmissionFractionParallel = NewType(
    'He3AnalyzerTransmissionFractionParallel', sc.DataArray
)
"""Transmission fraction of analyzer with parallel polarized incoming beam"""


He3AnalyzerTransmissionFractionAntiParallel = NewType(
    'He3AnalyzerTransmissionFractionAntiParallel', sc.DataArray
)
"""Transmission fraction of analyzer with anti-parallel polarized incoming beam"""


class He3TransmissionEmptyGlass(sl.Scope[PolarizingElement, sc.Variable], sc.Variable):
    """Transmission of the empty glass for a given cell."""


class He3DirectBeam(
    sl.ScopeTwoParams[PolarizingElement, PolarizationState, sc.DataArray], sc.DataArray
):
    """
    Direct beam data for a given cell and spin state as a function of wavelength.
    """


class He3Opacity0(sl.Scope[PolarizingElement, sc.Variable], sc.Variable):
    """Opacity at 1 Angstrom for a given cell."""


class He3OpacityFunction(Generic[PolarizingElement]):
    """Wavelength-dependent opacity function for a given cell."""

    def __init__(self, opacity0: sc.Variable):
        self._opacity0 = opacity0.to(unit='1/Angstrom')

    @property
    def opacity0(self) -> sc.Variable:
        return self._opacity0

    def __call__(self, wavelength: sc.Variable) -> sc.Variable:
        scale = broadcast_with_upper_bound_variances(
            self.opacity0, prototype=wavelength
        )
        return sc.DataArray(
            (scale * wavelength).to(unit='', copy=False),
            coords={'wavelength': wavelength},
        )


def he3_opacity_from_cell_params(
    pressure: He3CellPressure[PolarizingElement],
    length: He3CellLength[PolarizingElement],
    temperature: He3CellTemperature[PolarizingElement],
) -> He3Opacity0[PolarizingElement]:
    """Opacity 0 for a given cell, estimated from pressure and cell length."""
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
    return He3Opacity0[PolarizingElement](opacity0)


def he3_opacity_function_from_cell_opacity(
    opacity0: He3Opacity0[PolarizingElement],
) -> He3OpacityFunction[PolarizingElement]:
    """
    Opacity function for a given cell, based on pressure and cell length.

    Note that this can alternatively be defined via neutron beam data, see
    :py:func:`he3_opacity_function_from_beam_data`.
    """
    return He3OpacityFunction[PolarizingElement](opacity0)


def _with_midpoints(data: sc.DataArray, dim: str) -> sc.DataArray:
    if data.coords.is_edges(dim):
        return data.assign_coords({dim: sc.midpoints(data.coords[dim])})
    return data


def he3_opacity_function_from_beam_data(
    transmission_empty_glass: He3TransmissionEmptyGlass[PolarizingElement],
    transmission_fraction: He3CellTransmissionFractionIncomingUnpolarized[
        PolarizingElement, Depolarized
    ],
    opacity0_initial_guess: He3Opacity0[PolarizingElement],
) -> He3OpacityFunction[PolarizingElement]:
    """
    Opacity function for a given cell, based on direct beam data.

    Note that this can alternatively be defined via cell parameters, see
    :py:func:`he3_opacity_function_from_cell_opacity`. The cell opacity is used as an
    initial guess for the fit.
    """

    # TODO Fit the exponent, since too much weight on low wavelengths?
    def intensity(wavelength: sc.Variable, opacity0: sc.Variable) -> sc.Variable:
        opacity = He3OpacityFunction[PolarizingElement](opacity0)
        return transmission_empty_glass * sc.exp(-opacity(wavelength))

    popt, _ = sc.curve_fit(
        ['wavelength'],
        intensity,
        _with_midpoints(transmission_fraction, 'wavelength'),
        p0={'opacity0': opacity0_initial_guess},
    )
    return He3OpacityFunction[PolarizingElement](sc.values(popt['opacity0']).data)


class He3PolarizationFunction(Generic[PolarizingElement]):
    """Time-dependent polarization function for a given cell."""

    def __init__(self, C: sc.Variable, T1: sc.Variable):
        self._C = C
        self._T1 = T1

    @property
    def C(self) -> sc.Variable:
        return self._C

    @property
    def T1(self) -> sc.Variable:
        return self._T1

    def __call__(self, time: sc.Variable) -> sc.Variable:
        return sc.DataArray(self.C * sc.exp(-time / self.T1), coords={'time': time})


@dataclass
class He3TransmissionFunction(TransmissionFunction[PolarizingElement]):
    """Wavelength- and time-dependent transmission for a given cell."""

    opacity_function: He3OpacityFunction[PolarizingElement]
    polarization_function: He3PolarizationFunction[PolarizingElement]
    transmission_empty_glass: He3TransmissionEmptyGlass[PolarizingElement]

    def __call__(
        self, *, time: sc.Variable, wavelength: sc.Variable, plus_minus: PlusMinus
    ) -> sc.Variable:
        opacity = self.opacity_function(wavelength)
        polarization = self.polarization_function(time)
        if plus_minus == 'plus':
            polarization *= -1.0
        elif isinstance(plus_minus, sc.Variable):
            polarization *= -plus_minus
        return self.transmission_empty_glass * sc.exp(-opacity * (1.0 + polarization))

    def apply(self, data: sc.DataArray, plus_minus: PlusMinus) -> sc.DataArray:
        return self(
            time=data.coords['time'],
            wavelength=data.coords['wavelength'],
            plus_minus=plus_minus,
        )


def transmission_incoming_unpolarized(
    *,
    transmission_empty_glass: sc.Variable,
    opacity: sc.Variable,
    polarization: sc.Variable,
) -> sc.Variable:
    return transmission_empty_glass * sc.exp(-opacity) * sc.cosh(opacity * polarization)


def compute_transmission_fraction_from_direct_beam(
    direct_beam_no_cell: DirectBeamNoCell,
    direct_beam_polarized: He3DirectBeam[PolarizingElement, PolarizationState],
) -> He3CellTransmissionFractionIncomingUnpolarized[
    PolarizingElement, PolarizationState
]:
    """
    Compute the transmission fraction for a given cell and polarization state.

    This is defined as the ratio of the direct beam with the cell to the direct beam
    without the cell. The result is a function of wavelength and time.

    Note that this is possible only if the main detector is used to measure direct
    beam data. If direct beam is computed from monitors then, e.g., the SANS
    transmission fraction (as computed by a regular SANS workflow) should be used
    directly. Note that the regular SANS workflow also normalized to an empty beam
    run, make sure to not perform the division twice.
    """
    return He3CellTransmissionFractionIncomingUnpolarized[
        PolarizingElement, PolarizationState
    ](direct_beam_polarized / direct_beam_no_cell)


def get_he3_transmission_incoming_unpolarized_from_fit_to_direct_beam(
    transmission_fraction: He3CellTransmissionFractionIncomingUnpolarized[
        PolarizingElement, Polarized
    ],
    opacity_function: He3OpacityFunction[PolarizingElement],
    transmission_empty_glass: He3TransmissionEmptyGlass[PolarizingElement],
) -> TransmissionFunction[PolarizingElement]:
    """
    Transmission function for a given cell, with unpolarized incoming beam.

    This is composed from the opacity-function and the polarization-function.
    The implementation fits a time- and wavelength-dependent equation and returns
    the fitted T(t, lambda).

    DB_pol/DB = T_E * cosh(O(lambda)*P(t))*exp(-O(lambda))
    """

    def expected_transmission(
        wavelength: sc.Variable, time: sc.Variable, C: sc.Variable, T1: sc.Variable
    ) -> sc.Variable:
        polarization_function = He3PolarizationFunction[PolarizingElement](C=C, T1=T1)
        opacity = opacity_function(wavelength)
        polarization = polarization_function(time)
        return transmission_incoming_unpolarized(
            transmission_empty_glass=transmission_empty_glass,
            opacity=opacity,
            polarization=polarization,
        )

    popt, _ = sc.curve_fit(
        ['wavelength', 'time'],
        expected_transmission,
        _with_midpoints(transmission_fraction, 'wavelength'),
        p0={'C': sc.scalar(0.8, unit=''), 'T1': sc.scalar(400000.0, unit='s')},
    )
    # TODO Consider including variances from fit
    polarization_function = He3PolarizationFunction[PolarizingElement](
        C=sc.values(popt['C']).data, T1=sc.values(popt['T1']).data
    )
    return He3TransmissionFunction[PolarizingElement](
        opacity_function=opacity_function,
        polarization_function=polarization_function,
        transmission_empty_glass=transmission_empty_glass,
    )


def transmission_fraction_analyzer_parallel(
    upup: He3AnalyzerTransmissionFractionIncomingPolarized[Up, Up],
    downdown: He3AnalyzerTransmissionFractionIncomingPolarized[Down, Down],
) -> He3AnalyzerTransmissionFractionParallel:
    """
    Analyzer Transmission fraction with polarization parallel to incoming beam.

    It may not always we desirable to use both up-up and down-down transmission,
    fractions. If that is the case, set He3AnalyzerTransmissionFractionPlus directly
    instead of using this helper.
    """
    # The transmission fractions do not share a common time coordinate. Therefore,
    # we cannot concat along a third dimension for the fit, but concat along time,
    # with an additional coordinate for whether polarizations are parallel or
    # antiparallel.
    return sc.concat([upup, downdown], 'time').assign_coords(plus_minus=sc.scalar(1))


def transmission_fraction_analyzer_antiparallel(
    updown: He3AnalyzerTransmissionFractionIncomingPolarized[Up, Down],
    downup: He3AnalyzerTransmissionFractionIncomingPolarized[Down, Up],
) -> He3AnalyzerTransmissionFractionAntiParallel:
    """
    Analyzer transmission fraction with polarization anti-parallel to incoming beam.

    It may not always we desirable to use both up-down and down-up transmission,
    fractions. If that is the case, set He3AnalyzerTransmissionFractionMinus directly
    instead of using this helper.
    """
    # The transmission fractions do not share a common time coordinate. Therefore,
    # we cannot concat along a third dimension for the fit, but concat along time,
    # with an additional coordinate for whether polarizations are parallel or
    # antiparallel.
    return sc.concat([updown, downup], 'time').assign_coords(plus_minus=sc.scalar(-1))


def get_he3_transmission_incoming_polarized_from_fit_to_direct_beam(
    plus: He3AnalyzerTransmissionFractionParallel,
    minus: He3AnalyzerTransmissionFractionAntiParallel,
    opacity_function: He3OpacityFunction[Analyzer],
    transmission_empty_glass: He3TransmissionEmptyGlass[Analyzer],
) -> TransmissionFunction[Analyzer]:
    """
    Transmission function for the analyzer, computed with incoming polarized beam.

    This is composed from the opacity-function and the polarization-function.
    The implementation fits a time- and wavelength-dependent equation and returns
    the fitted T(t, lambda).
    """
    if (plus_minus := plus.coords.get('plus_minus')) is not None:
        if not sc.all(plus_minus == sc.scalar(1)):
            raise ValueError('Expected plus-minus coordinate of plus channel to be +1.')
    else:
        plus = plus.assign_coords(plus_minus=sc.scalar(1))
    if (plus_minus := minus.coords.get('plus_minus')) is not None:
        if not sc.all(plus_minus == sc.scalar(-1)):
            raise ValueError(
                'Expected plus-minus coordinate of minus channel to be -1.'
            )
    else:
        minus = minus.assign_coords(plus_minus=sc.scalar(-1))

    transmission_fraction = sc.concat([plus, minus], 'time')

    def expected_transmission(
        wavelength: sc.Variable,
        time: sc.Variable,
        plus_minus: sc.Variable,
        C: sc.Variable,
        T1: sc.Variable,
    ) -> sc.Variable:
        polarization_function = He3PolarizationFunction[PolarizingElement](C=C, T1=T1)
        return He3TransmissionFunction(
            opacity_function=opacity_function,
            polarization_function=polarization_function,
            transmission_empty_glass=transmission_empty_glass,
        )(time=time, wavelength=wavelength, plus_minus=plus_minus)

    popt, _ = sc.curve_fit(
        ['wavelength', 'time', 'plus_minus'],
        expected_transmission,
        _with_midpoints(transmission_fraction, 'wavelength'),
        p0={'C': sc.scalar(0.8, unit=''), 'T1': sc.scalar(400000.0, unit='s')},
    )
    # TODO Consider including variances from fit
    polarization_function = He3PolarizationFunction[PolarizingElement](
        C=sc.values(popt['C']).data, T1=sc.values(popt['T1']).data
    )
    return He3TransmissionFunction[PolarizingElement](
        opacity_function=opacity_function,
        polarization_function=polarization_function,
        transmission_empty_glass=transmission_empty_glass,
    )


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


DirectBeamQRange = NewType('DirectBeamQRange', sc.Variable)
"""Q-range defining the direct beam region in a direct beam measurement."""

DirectBeamBackgroundQRange = NewType('DirectBeamBackgroundQRange', sc.Variable)
"""Q-range defining the direct beam background region in a direct beam measurement."""

ReducedDirectBeamDataNoCell = NewType('ReducedDirectBeamDataNoCell', sc.DataArray)


class ReducedDirectBeamData(
    sl.ScopeTwoParams[PolarizingElement, PolarizationState, sc.DataArray], sc.DataArray
):
    """Direct beam data for a given cell, as a function of wavelength and time."""


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
    data: ReducedDirectBeamData[PolarizingElement, PolarizationState],
    q_range: DirectBeamQRange,
    background_q_range: DirectBeamBackgroundQRange,
) -> He3DirectBeam[PolarizingElement, PolarizationState]:
    """
    Returns the direct beam function for a given cell.

    The result is background-subtracted and returned as function of wavelength and
    wall-clock time. The time dependence is coarse, i.e., due to different time
    intervals at which the direct beam is measured.
    """
    return He3DirectBeam[PolarizingElement, PolarizationState](
        compute_direct_beam(
            data=data,
            q_range=q_range,
            background_q_range=background_q_range,
        )
    )


providers = (he3_opacity_from_cell_params, direct_beam, direct_beam_with_cell)


def He3CellWorkflow(
    *, in_situ: bool = True, incoming_polarized: bool = False
) -> sl.Pipeline:
    """
    Workflow for computing transmission functions for He3 cells.

    This can handle polarizers as well as analyzers.

    Parameters
    ----------
    in_situ :
        Whether to use an in-situ definition of the cell opacity based on cell
        parameters, or an ex-situ definition based on direct beam data. The latter
        requires a direct-beam measurement with depolarized cells.
    incoming_polarized :
        Whether the incoming beam for computing the analyzer transmission is polarized.
        This is the case in beamlines with a supermirror polarizer, but also if the
        polarizer is not removed from the beam during the analyzer transmission
        measurement.
    """
    workflow = sl.Pipeline(providers)
    if in_situ:
        workflow.insert(he3_opacity_function_from_cell_opacity)
    else:
        workflow.insert(he3_opacity_function_from_beam_data)
    # Note that the incoming-unpolarized function is inserted even if
    # incoming_polarized=True, since the incoming-unpolarized function is still
    # required for computing the *polarizer* transmission calculation.
    workflow.insert(get_he3_transmission_incoming_unpolarized_from_fit_to_direct_beam)
    if incoming_polarized:
        workflow.insert(transmission_fraction_analyzer_parallel)
        workflow.insert(transmission_fraction_analyzer_antiparallel)
        workflow.insert(get_he3_transmission_incoming_polarized_from_fit_to_direct_beam)
    return workflow
