# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
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
    return PolarizationCorrection[Up, Up](
        upup=polarizer.diag * analyzer.diag,
        updown=polarizer.diag * analyzer.off_diag,
        downup=polarizer.off_diag * analyzer.diag,
        downdown=polarizer.off_diag * analyzer.off_diag,
    )


def compute_polarization_correction_updown(
    analyzer: PolarizingElementCorrection[Up, Down, Analyzer],
    polarizer: PolarizingElementCorrection[Up, Down, Polarizer],
) -> PolarizationCorrection[Up, Down]:
    """
    Up-down column of combined correction coefficients for polarizer and analyzer.

    See :py:func:`compute_polarization_correction_upup` for more details.
    """
    return PolarizationCorrection[Up, Down](
        upup=polarizer.diag * analyzer.off_diag,
        updown=polarizer.diag * analyzer.diag,
        downup=polarizer.off_diag * analyzer.off_diag,
        downdown=polarizer.off_diag * analyzer.diag,
    )


def compute_polarization_correction_downup(
    analyzer: PolarizingElementCorrection[Down, Up, Analyzer],
    polarizer: PolarizingElementCorrection[Down, Up, Polarizer],
) -> PolarizationCorrection[Down, Up]:
    """
    Down-up column of combined correction coefficients for polarizer and analyzer.

    See :py:func:`compute_polarization_correction_upup` for more details.
    """
    return PolarizationCorrection[Down, Up](
        upup=polarizer.off_diag * analyzer.diag,
        updown=polarizer.off_diag * analyzer.off_diag,
        downup=polarizer.diag * analyzer.diag,
        downdown=polarizer.diag * analyzer.off_diag,
    )


def compute_polarization_correction_downdown(
    analyzer: PolarizingElementCorrection[Down, Down, Analyzer],
    polarizer: PolarizingElementCorrection[Down, Down, Polarizer],
) -> PolarizationCorrection[Down, Down]:
    """
    Down-down column of combined correction coefficients for polarizer and analyzer.

    See :py:func:`compute_polarization_correction_upup` for more details.
    """
    return PolarizationCorrection[Down, Down](
        upup=polarizer.off_diag * analyzer.off_diag,
        updown=polarizer.off_diag * analyzer.diag,
        downup=polarizer.diag * analyzer.off_diag,
        downdown=polarizer.diag * analyzer.diag,
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
    return sciline.Pipeline(
        (
            compute_polarizing_element_correction,
            compute_polarization_correction_upup,
            compute_polarization_correction_updown,
            compute_polarization_correction_downup,
            compute_polarization_correction_downdown,
            compute_polarization_corrected_data,
        )
    )
