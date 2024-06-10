# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Callable, Generic

import sciline
import scipp as sc

from .types import (
    Analyzer,
    AnalyzerSpin,
    Down,
    PolarizationCorrectedData,
    PolarizationCorrection,
    Polarizer,
    PolarizerSpin,
    PolarizingElement,
    PolarizingElementCorrection,
    ReducedSampleDataBySpinChannel,
    TransmissionFunction,
    Up,
)


def compute_polarizing_element_correction(
    channel: ReducedSampleDataBySpinChannel[PolarizerSpin, AnalyzerSpin],
    transmission: TransmissionFunction[PolarizingElement],
) -> PolarizingElementCorrection[PolarizerSpin, AnalyzerSpin, PolarizingElement]:
    """
    Compute matrix coefficients for the correction of a polarizing element.

    The coefficients stem from the inverse of a symmetric matrix of the form
    [[Tplus, Tminus], [Tminus, Tplus]]. The inverse is given by a matrix
        mat = 1/denom * [[Tplus, -Tminus], [-Tminus, Tplus]],
    with
        denom = Tplus**2 - Tminus**2.
    As there are only two unique elements in the matrix, we return them as a dataclass
    with diagonal and off-diagonal elements.

    Parameters
    ----------
    channel :
        Data including wavelength (and time) for a given spin channel.
    transmission :
        Transmission function for the polarizing element.

    Returns
    -------
    :
        Correction matrix coefficients.

    """
    t_plus = transmission.apply(channel, 'plus')
    t_minus = transmission.apply(channel, 'minus')
    t_minus *= -1
    denom = t_plus**2 - t_minus**2
    sc.reciprocal(denom, out=denom)
    t_plus *= denom
    t_minus *= denom
    return PolarizingElementCorrection[PolarizerSpin, AnalyzerSpin, PolarizingElement](
        diag=t_plus, off_diag=t_minus
    )


class AnalyzerFlipper(Generic[AnalyzerSpin]):
    """Flipper for the analyzer."""

    def __call__(
        self, up: sc.Variable, down: sc.Variable
    ) -> tuple[sc.Variable, sc.Variable]:
        """Flip the analyzer."""
        raise NotImplementedError


class PolarizerFlipper(Generic[PolarizerSpin]):
    """Flipper for the polarizer."""

    def __call__(
        self, up: sc.Variable, down: sc.Variable
    ) -> tuple[sc.Variable, sc.Variable]:
        """Flip the polarizer."""
        raise NotImplementedError


def no_flipper(up: sc.Variable, down: sc.Variable) -> tuple[sc.Variable, sc.Variable]:
    return up, down


def flip(up: sc.Variable, down: sc.Variable) -> tuple[sc.Variable, sc.Variable]:
    return down, up


Flipper = Callable[[sc.Variable, sc.Variable], tuple[sc.Variable, sc.Variable]]


def _compute_pol_corr(
    *,
    analyzer: PolarizingElementCorrection[PolarizerSpin, AnalyzerSpin, Analyzer],
    polarizer: PolarizingElementCorrection[PolarizerSpin, AnalyzerSpin, Polarizer],
    analyzer_up: bool,
    polarizer_up: bool,
    analyzer_flipper: Flipper = no_flipper,
    polarizer_flipper: Flipper = no_flipper,
) -> PolarizationCorrection[PolarizerSpin, AnalyzerSpin]:
    """
    Compute a column of the combined correction coefficients for polarizer and analyzer.

    This is effectively a column resulting from a sparse matrix-matrix product.

    Parameters
    ----------
    analyzer :
        Correction coefficients for the analyzer.
    polarizer :
        Correction coefficients for the polarizer.
    analyzer_up :
        Whether the analyzer is in the up position.
    polarizer_up :
        Whether the polarizer is in the up position.

    Returns
    -------
    :
        Combined correction coefficients.
    """

    # Can the swap performed by the `get` method be handled using a fake flipper?
    a_up, a_down = analyzer_flipper(*analyzer.get(up=analyzer_up))
    p_up, p_down = polarizer.get(up=polarizer_up)
    # upup = p_up * a_up
    # updown = p_up * a_down
    # downup = p_down * a_up
    # downdown = p_down * a_down

    # for polarizer-up, polarized flipper scales second with (1-1/f)
    # upup, downup = polarizer_flipper(upup, downup)
    # updown, downdown = polarizer_flipper(updown, downdown)

    return PolarizationCorrection[PolarizerSpin, AnalyzerSpin](
        upup=p_up * a_up,
        updown=p_up * a_down,
        downup=p_down * a_up,
        downdown=p_down * a_down,
    )


def compute_polarization_correction(
    *,
    analyzer: PolarizingElementCorrection[PolarizerSpin, AnalyzerSpin, Analyzer],
    polarizer: PolarizingElementCorrection[PolarizerSpin, AnalyzerSpin, Polarizer],
    analyzer_flipper: AnalyzerFlipper[AnalyzerSpin],
    polarizer_flipper: PolarizerFlipper[PolarizerSpin],
) -> PolarizationCorrection[PolarizerSpin, AnalyzerSpin]:
    a_up, a_down = analyzer_flipper(analyzer.diag, analyzer.off_diag)
    p_up, p_down = polarizer.diag, polarizer.off_diag
    upup = p_up * a_up
    updown = p_up * a_down
    downup = p_down * a_up
    downdown = p_down * a_down

    # for polarizer-up, polarized flipper scales second with (1-1/f)
    upup, downup = polarizer_flipper(upup, downup)
    updown, downdown = polarizer_flipper(updown, downdown)
    return PolarizationCorrection[PolarizerSpin, AnalyzerSpin](
        upup=upup,
        updown=updown,
        downup=downup,
        downdown=downdown,
    )


# TODO remove this
def compute_polarization_correction_upup(
    analyzer: PolarizingElementCorrection[Up, Up, Analyzer],
    polarizer: PolarizingElementCorrection[Up, Up, Polarizer],
) -> PolarizationCorrection[Up, Up]:
    """
    Up-up column of combined correction coefficients for polarizer and analyzer.

    This is effectively a column resulting from a sparse matrix-matrix product.

    This function returns the up-up column.

    Parameters
    ----------
    analyzer :
        Correction coefficients for the analyzer.
    polarizer :
        Correction coefficients for the polarizer.

    Returns
    -------
    :
        Combined correction coefficients.
    """
    return _compute_pol_corr(
        analyzer=analyzer, polarizer=polarizer, analyzer_up=True, polarizer_up=True
    )


def compute_polarization_correction_updown(
    analyzer: PolarizingElementCorrection[Up, Down, Analyzer],
    polarizer: PolarizingElementCorrection[Up, Down, Polarizer],
) -> PolarizationCorrection[Up, Down]:
    """
    Up-down column of combined correction coefficients for polarizer and analyzer.

    See :py:func:`compute_polarization_correction_upup` for more details.
    """
    return _compute_pol_corr(
        analyzer=analyzer, polarizer=polarizer, analyzer_up=False, polarizer_up=True
    )


def compute_polarization_correction_downup(
    analyzer: PolarizingElementCorrection[Down, Up, Analyzer],
    polarizer: PolarizingElementCorrection[Down, Up, Polarizer],
) -> PolarizationCorrection[Down, Up]:
    """
    Down-up column of combined correction coefficients for polarizer and analyzer.

    See :py:func:`compute_polarization_correction_upup` for more details.
    """
    return _compute_pol_corr(
        analyzer=analyzer, polarizer=polarizer, analyzer_up=True, polarizer_up=False
    )


def compute_polarization_correction_downdown(
    analyzer: PolarizingElementCorrection[Down, Down, Analyzer],
    polarizer: PolarizingElementCorrection[Down, Down, Polarizer],
) -> PolarizationCorrection[Down, Down]:
    """
    Down-down column of combined correction coefficients for polarizer and analyzer.

    See :py:func:`compute_polarization_correction_upup` for more details.
    """
    return _compute_pol_corr(
        analyzer=analyzer, polarizer=polarizer, analyzer_up=False, polarizer_up=False
    )


def compute_polarization_corrected_data(
    channel: ReducedSampleDataBySpinChannel[PolarizerSpin, AnalyzerSpin],
    polarization_correction: PolarizationCorrection[PolarizerSpin, AnalyzerSpin],
) -> PolarizationCorrectedData[PolarizerSpin, AnalyzerSpin]:
    # TODO Would like to use inplace ops, but modifying input is dodgy. Maybe combine
    # into a single function?
    return PolarizationCorrectedData(
        upup=channel * polarization_correction.upup,
        updown=channel * polarization_correction.updown,
        downup=channel * polarization_correction.downup,
        downdown=channel * polarization_correction.downdown,
    )


def CorrectionWorkflow() -> sciline.Pipeline:
    workflow = sciline.Pipeline(
        (
            compute_polarizing_element_correction,
            compute_polarization_correction,
            # compute_polarization_correction_upup,
            # compute_polarization_correction_updown,
            # compute_polarization_correction_downup,
            # compute_polarization_correction_downdown,
            compute_polarization_corrected_data,
        )
    )
    # This "flipper" setup represents the matrix structure, no actual flippers
    workflow[AnalyzerFlipper[Up]] = no_flipper
    workflow[AnalyzerFlipper[Down]] = flip
    workflow[PolarizerFlipper[Up]] = no_flipper
    workflow[PolarizerFlipper[Down]] = flip
    return workflow
