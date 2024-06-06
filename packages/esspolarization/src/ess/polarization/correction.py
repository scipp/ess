# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sciline
import scipp as sc

from .types import (
    Analyzer,
    AnalyzerCorrectedData,
    Down,
    PolarizationCorrectedData,
    Polarizer,
    PolarizerSpin,
    PolarizingElement,
    ReducedSampleDataBySpinChannel,
    TransmissionFunction,
    Up,
)


def correct_for_polarizing_element(
    up: sc.DataArray,
    down: sc.DataArray,
    transmission_function: TransmissionFunction[PolarizingElement],
) -> tuple[sc.DataArray, sc.DataArray]:
    """
    denom = Tplus**2 - Tminus**2
    mat = [[Tplus, -Tminus], [-Tminus, Tplus]]
    """
    t_plus_up = transmission_function.apply(up, 'plus')
    t_minus_up = -transmission_function.apply(up, 'minus')
    t_plus_down = transmission_function.apply(down, 'plus')
    t_minus_down = -transmission_function.apply(down, 'minus')
    up = up / (t_plus_up**2 - t_minus_up**2)
    down = down / (t_plus_down**2 - t_minus_down**2)
    t_plus_up *= up
    t_minus_up *= up
    t_plus_down *= down
    t_minus_down *= down
    out_up = sc.concat([t_plus_up, t_minus_down], up.dim)
    out_down = sc.concat([t_minus_up, t_plus_down], down.dim)
    return out_up, out_down


def correct_for_analyzer(
    up: ReducedSampleDataBySpinChannel[PolarizerSpin, Up],
    down: ReducedSampleDataBySpinChannel[PolarizerSpin, Down],
    transmission: TransmissionFunction[Analyzer],
) -> AnalyzerCorrectedData[PolarizerSpin]:
    up, down = correct_for_polarizing_element(up, down, transmission)
    return AnalyzerCorrectedData(up=up, down=down)


def correct_for_polarizer(
    up: AnalyzerCorrectedData[Up],
    down: AnalyzerCorrectedData[Down],
    transmission: TransmissionFunction[Polarizer],
) -> PolarizationCorrectedData:
    upup, downup = correct_for_polarizing_element(up.up, down.up, transmission)
    updown, downdown = correct_for_polarizing_element(up.down, down.down, transmission)
    return PolarizationCorrectedData(
        upup=upup, updown=updown, downup=downup, downdown=downdown
    )


def CorrectionWorkflow() -> sciline.Pipeline:
    return sciline.Pipeline((correct_for_analyzer, correct_for_polarizer))
