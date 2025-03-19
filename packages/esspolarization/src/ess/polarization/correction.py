# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import Generic

import sciline
import scipp as sc

from .types import (
    Analyzer,
    AnalyzerSpin,
    Down,
    FlipperEfficiency,
    HalfPolarizedCorrectedData,
    HalfPolarizedCorrection,
    NoAnalyzer,
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


@dataclass
class InverseFlipperMatrix(Generic[PolarizerSpin, PolarizingElement]):
    """Flipper matrix, combined with component flip for down component"""

    efficiency: FlipperEfficiency[PolarizingElement]
    swap: bool

    def from_left(
        self, up: sc.Variable, down: sc.Variable
    ) -> tuple[sc.Variable, sc.Variable]:
        """Apply inverse flipper matrix from the left (for analyzer)"""
        if self.swap:
            up, down = down, up
        f = 1 / self.efficiency.value
        if f == 1:
            return up, down
        return up, (1 - f) * up + f * down

    def from_right(
        self, up: sc.Variable, down: sc.Variable
    ) -> tuple[sc.Variable, sc.Variable]:
        """Apply inverse flipper matrix from the right (for polarizer)"""
        f = 1 / self.efficiency.value
        if f == 1:
            return (down, up) if self.swap else (up, down)
        if self.swap:
            return f * down, f * up
        else:
            return up + (1 - f) * down, down + (1 - f) * up


def make_spin_flipping_matrix_up(
    efficiency: FlipperEfficiency[PolarizingElement],
) -> InverseFlipperMatrix[Up, PolarizingElement]:
    return InverseFlipperMatrix[Up, PolarizingElement](
        efficiency=efficiency, swap=False
    )


def make_spin_flipping_matrix_down(
    efficiency: FlipperEfficiency[PolarizingElement],
) -> InverseFlipperMatrix[Down, PolarizingElement]:
    return InverseFlipperMatrix[Down, PolarizingElement](
        efficiency=efficiency, swap=True
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
        Data including wavelength (and time) for a given spin channel. Note that the
        values are not actually used here, but the data's coordinates are required for
        evaluating the transmission function.
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


def compute_polarization_correction(
    *,
    analyzer: PolarizingElementCorrection[PolarizerSpin, AnalyzerSpin, Analyzer],
    polarizer: PolarizingElementCorrection[PolarizerSpin, AnalyzerSpin, Polarizer],
    analyzer_flipper: InverseFlipperMatrix[AnalyzerSpin, Analyzer],
    polarizer_flipper: InverseFlipperMatrix[PolarizerSpin, Polarizer],
) -> PolarizationCorrection[PolarizerSpin, AnalyzerSpin]:
    """
    Compute columns of combined correction coefficients for polarizer and analyzer.

    This is effectively a column resulting from a sparse matrix-matrix product.

    Parameters
    ----------
    analyzer :
        Correction coefficients for the analyzer.
    polarizer :
        Correction coefficients for the polarizer.
    analyzer_flipper :
        Flipper matrix for the analyzer.
    polarizer_flipper :
        Flipper matrix for the polarizer.

    Returns
    -------
    :
        Combined correction coefficients.
    """
    a_up, a_down = analyzer_flipper.from_left(analyzer.diag, analyzer.off_diag)
    p_up, p_down = polarizer_flipper.from_right(polarizer.diag, polarizer.off_diag)
    return PolarizationCorrection[PolarizerSpin, AnalyzerSpin](
        upup=p_up * a_up,
        updown=p_up * a_down,
        downup=p_down * a_up,
        downdown=p_down * a_down,
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


def compute_half_polarized_correction(
    *,
    polarizer: PolarizingElementCorrection[PolarizerSpin, NoAnalyzer, Polarizer],
    polarizer_flipper: InverseFlipperMatrix[PolarizerSpin, Polarizer],
) -> HalfPolarizedCorrection[PolarizerSpin]:
    p_up, p_down = polarizer_flipper.from_right(polarizer.diag, polarizer.off_diag)
    return HalfPolarizedCorrection[PolarizerSpin](up=p_up, down=p_down)


def compute_half_polarized_corrected_data(
    channel: ReducedSampleDataBySpinChannel[PolarizerSpin, NoAnalyzer],
    polarization_correction: HalfPolarizedCorrection[PolarizerSpin],
) -> HalfPolarizedCorrectedData[PolarizerSpin]:
    return HalfPolarizedCorrectedData(
        up=channel * polarization_correction.up,
        down=channel * polarization_correction.down,
    )


def CorrectionWorkflow(half_polarized: bool = False) -> sciline.Pipeline:
    """
    Create a workflow for polarization correction.

    This is a basic workflow that requires setting the transmission function directly.
    See :py:func:`PolarizationAnalysisWorkflow` and :py:func:`HalfPolarizedWorkflow` for
    workflows that can compute the transmission function from a polarizer or analyzer.

    Parameters
    ----------
    half_polarized :
        If True, the workflow is for a half-polarized case (polarizer only).
        If False, the workflow is for a full polarization case (polarizer and
        analyzer).

    See Also
    --------
    PolarizationAnalysisWorkflow
    HalfPolarizedWorkflow
    """
    workflow = sciline.Pipeline(
        (
            make_spin_flipping_matrix_up,
            make_spin_flipping_matrix_down,
            compute_polarizing_element_correction,
        )
    )
    if half_polarized:
        workflow.insert(compute_half_polarized_correction)
        workflow.insert(compute_half_polarized_corrected_data)
    else:
        workflow.insert(compute_polarization_correction)
        workflow.insert(compute_polarization_corrected_data)
    # If there is no flipper, setting an efficiency of 1.0 is equivalent to not using
    # a flipper.
    workflow[FlipperEfficiency[PolarizingElement]] = FlipperEfficiency[
        PolarizingElement
    ](value=1.0)
    return workflow


def PolarizationAnalysisWorkflow(
    *,
    polarizer_workflow: sciline.Pipeline,
    analyzer_workflow: sciline.Pipeline,
) -> sciline.Pipeline:
    """
    Create a polarization analysis workflow.

    Parameters
    ----------
    polarizer_workflow :
        Workflow for the polarizer, e.g., a He3CellWorkflow or SupermirrorWorkflow.
    analyzer_workflow :
        Workflow for the analyzer, e.g., a He3CellWorkflow or SupermirrorWorkflow.

    Returns
    -------
    :
        Full workflow for polarization analysis.
    """

    workflow = CorrectionWorkflow()
    workflow[TransmissionFunction[Polarizer]] = polarizer_workflow[
        TransmissionFunction[Polarizer]
    ]
    workflow[TransmissionFunction[Analyzer]] = analyzer_workflow[
        TransmissionFunction[Analyzer]
    ]
    return workflow


def HalfPolarizedWorkflow(
    *,
    polarizer_workflow: sciline.Pipeline,
) -> sciline.Pipeline:
    """
    Create a half-polarized workflow, i.e, with a polarizer but no analyzer.

    Parameters
    ----------
    polarizer_workflow :
        Workflow for the polarizer, e.g., a He3CellWorkflow or SupermirrorWorkflow.

    Returns
    -------
    :
        Half-polarized workflow.
    """
    workflow = CorrectionWorkflow(half_polarized=True)
    workflow[TransmissionFunction[Polarizer]] = polarizer_workflow[
        TransmissionFunction[Polarizer]
    ]
    return workflow
