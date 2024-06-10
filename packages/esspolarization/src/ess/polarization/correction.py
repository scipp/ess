# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass

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


@dataclass
class CorrectionComponents:
    diag: sc.DataArray
    off_diag: sc.DataArray


def compute_correction_from_component(
    channel: sc.DataArray,
    transmission: TransmissionFunction[PolarizingElement],
) -> CorrectionComponents:
    t_plus = transmission.apply(channel, 'plus')
    t_minus = -transmission.apply(channel, 'minus')
    base = channel / (t_plus**2 - t_minus**2)
    t_plus *= base
    t_minus *= base
    return CorrectionComponents(diag=t_plus, off_diag=t_minus)


def correct_for_polarizing_element(
    up: sc.DataArray,
    down: sc.DataArray,
    transmission_function: TransmissionFunction[PolarizingElement],
    prefix: str = '',
) -> tuple[sc.Dataset, sc.Dataset]:
    """
    denom = Tplus**2 - Tminus**2
    mat = [[Tplus, -Tminus], [-Tminus, Tplus]]
    """
    components = compute_correction_from_component(up, transmission_function)
    t_plus_up = components.diag
    t_minus_up = components.off_diag
    components = compute_correction_from_component(down, transmission_function)
    t_plus_down = components.diag
    t_minus_down = components.off_diag
    # We combine into Datasets so we can share coordinates when concatenating later
    # TODO I think Scipp does not actually does this, it has a naive impl.
    # Also, keep in mind that we are in general Q-binned, i.e., concat bins!
    return (
        sc.Dataset({f'{prefix}up': t_plus_up, f'{prefix}down': t_minus_up}),
        sc.Dataset({f'{prefix}up': t_minus_down, f'{prefix}down': t_plus_down}),
    )


def correct_for_analyzer(
    analyzer_up: ReducedSampleDataBySpinChannel[PolarizerSpin, Up],
    analyzer_down: ReducedSampleDataBySpinChannel[PolarizerSpin, Down],
    transmission: TransmissionFunction[Analyzer],
) -> AnalyzerCorrectedData[PolarizerSpin]:
    part1, part2 = correct_for_polarizing_element(
        analyzer_up, analyzer_down, transmission, prefix='analyzer_'
    )
    return AnalyzerCorrectedData[PolarizerSpin](
        **sc.concat([part1, part2], analyzer_up.dim)
    )


def correct_for_polarizer(
    polarizer_up: AnalyzerCorrectedData[Up],
    polarizer_down: AnalyzerCorrectedData[Down],
    transmission: TransmissionFunction[Polarizer],
) -> PolarizationCorrectedData:
    up_part1, up_part2 = correct_for_polarizing_element(
        polarizer_up.analyzer_up, polarizer_down.analyzer_up, transmission, prefix='up'
    )
    down_part1, down_part2 = correct_for_polarizing_element(
        polarizer_up.analyzer_down,
        polarizer_down.analyzer_down,
        transmission,
        prefix='down',
    )
    part1 = sc.Dataset(**up_part1, **down_part1)
    part2 = sc.Dataset(**up_part2, **down_part2)
    return PolarizationCorrectedData(
        **sc.concat([part1, part2], polarizer_up.analyzer_up.dim)
    )


def CorrectionWorkflow() -> sciline.Pipeline:
    return sciline.Pipeline((correct_for_analyzer, correct_for_polarizer))
