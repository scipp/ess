# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from scipp.testing import assert_allclose

from ess.polarization.correction import (
    CorrectionWorkflow,
    compute_polarizing_element_correction,
)
from ess.polarization.types import (
    Analyzer,
    Down,
    PolarizationCorrectedData,
    Polarizer,
    ReducedSampleDataBySpinChannel,
    TransmissionFunction,
    Up,
)


class SimpleTransmissionFunction:
    def __call__(
        self, time: sc.Variable, wavelength: sc.Variable, plus_minus: str
    ) -> sc.Variable:
        if plus_minus == 'plus':
            return 10 * time * (2 + wavelength)
        else:
            return 10 * time * (2 - wavelength)

    def apply(self, da: sc.DataArray, plus_minus: str) -> float:
        time = da.coords['time']
        wavelength = da.coords['wavelength']
        return self(time, wavelength, plus_minus)


def test_compute_polarizing_element_correction() -> None:
    time = sc.linspace('event', 1, 10, 10, unit='')
    wavelength = sc.linspace('event', 0.1, 1, 10, unit='')
    events = sc.DataArray(
        sc.arange('event', 10),
        coords={'time': time, 'wavelength': wavelength},
    )
    transmission = SimpleTransmissionFunction()

    result = compute_polarizing_element_correction(
        channel=events, transmission=transmission
    )
    diag = result.diag
    off_diag = result.off_diag
    assert diag.sizes == {'event': 10}
    assert off_diag.sizes == {'event': 10}
    transmission_plus = transmission(time, wavelength, 'plus')
    transmission_minus = transmission(time, wavelength, 'minus')
    denom = transmission_plus**2 - transmission_minus**2
    assert_allclose(diag, transmission_plus / denom)
    assert_allclose(off_diag, -transmission_minus / denom)


class FakeTransmissionFunction:
    def __init__(self, coeffs: np.ndarray) -> None:
        self.coeffs = coeffs

    def apply(self, _: sc.DataArray, plus_minus: str) -> float:
        if plus_minus == 'plus':
            return float(self.coeffs[0][0])
        else:
            return float(self.coeffs[0][1])


def test_correction_workflow_computes_and_applies_matrix_inverse() -> None:
    ground_truth = np.array([7.0, 11.0, 13.0, 17.0])
    analyzer = np.array([[1.3, 0.9], [0.9, 1.3]])
    polarizer = np.array([[1.1, 0.7], [0.7, 1.1]])
    identity = np.array([[1.0, 0.0], [0.0, 1.0]])
    input = np.kron(identity, analyzer) @ np.kron(polarizer, identity) @ ground_truth

    workflow = CorrectionWorkflow()
    workflow[TransmissionFunction[Analyzer]] = FakeTransmissionFunction(analyzer)
    workflow[TransmissionFunction[Polarizer]] = FakeTransmissionFunction(polarizer)
    workflow[ReducedSampleDataBySpinChannel[Up, Up]] = input[0]
    workflow[ReducedSampleDataBySpinChannel[Up, Down]] = input[1]
    workflow[ReducedSampleDataBySpinChannel[Down, Up]] = input[2]
    workflow[ReducedSampleDataBySpinChannel[Down, Down]] = input[3]

    result = np.zeros(4)
    for pol in [Up, Down]:
        for ana in [Up, Down]:
            contrib = workflow.compute(PolarizationCorrectedData[pol, ana])
            result += [contrib.upup, contrib.updown, contrib.downup, contrib.downdown]
    np.testing.assert_allclose(result, ground_truth)
