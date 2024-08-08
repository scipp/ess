# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import sciline
import scipp as sc
from scipp.testing import assert_allclose

import ess.polarization as pol
from ess.polarization import (
    CorrectionWorkflow,
    He3CellWorkflow,
    PolarizationAnalysisWorkflow,
    SupermirrorWorkflow,
)
from ess.polarization.correction import (
    FlipperEfficiency,
    compute_polarizing_element_correction,
)
from ess.polarization.types import (
    Analyzer,
    Down,
    HalfPolarizedCorrectedData,
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
            return sc.scalar(self.coeffs[0][0])
        else:
            return sc.scalar(self.coeffs[0][1])


def test_correction_workflow_computes_and_applies_matrix_inverse() -> None:
    ground_truth = np.array([7.0, 11.0, 13.0, 17.0])
    analyzer = np.array([[1.3, 0.9], [0.9, 1.3]])
    polarizer = np.array([[1.1, 0.7], [0.7, 1.1]])
    identity = np.array([[1.0, 0.0], [0.0, 1.0]])
    intensity = (
        np.kron(identity, analyzer) @ np.kron(polarizer, identity) @ ground_truth
    )

    workflow = CorrectionWorkflow()
    workflow[TransmissionFunction[Analyzer]] = FakeTransmissionFunction(analyzer)
    workflow[TransmissionFunction[Polarizer]] = FakeTransmissionFunction(polarizer)
    workflow[ReducedSampleDataBySpinChannel[Up, Up]] = intensity[0]
    workflow[ReducedSampleDataBySpinChannel[Up, Down]] = intensity[1]
    workflow[ReducedSampleDataBySpinChannel[Down, Up]] = intensity[2]
    workflow[ReducedSampleDataBySpinChannel[Down, Down]] = intensity[3]

    result = np.zeros(4)
    for pola in [Up, Down]:
        for ana in [Up, Down]:
            contrib = workflow.compute(PolarizationCorrectedData[pola, ana])
            contrib = sc.concat(
                [contrib.upup, contrib.updown, contrib.downup, contrib.downdown],
                'dummy',
            )
            result += contrib.values
    np.testing.assert_allclose(result, ground_truth)


@pytest.mark.parametrize('f', [0.1, 0.5, 0.9, 0.99, 1.0])
def test_workflow_with_analyzer_flipper_computes_and_applies_matrix_inverse(
    f: float,
) -> None:
    ground_truth = np.array([7.0, 11.0, 13.0, 17.0])
    analyzer = np.array([[1.3, 0.9], [0.9, 1.3]])
    polarizer = np.array([[1.1, 0.7], [0.7, 1.1]])
    identity = np.array([[1.0, 0.0], [0.0, 1.0]])

    flipper = np.array([[1.0, 0.0], [1 - f, f]])
    intensity = (
        np.kron(identity, analyzer @ flipper)
        @ np.kron(polarizer, identity)
        @ ground_truth
    )

    workflow = CorrectionWorkflow()
    workflow[TransmissionFunction[Analyzer]] = FakeTransmissionFunction(analyzer)
    workflow[TransmissionFunction[Polarizer]] = FakeTransmissionFunction(polarizer)
    workflow[ReducedSampleDataBySpinChannel[Up, Up]] = intensity[0]
    workflow[ReducedSampleDataBySpinChannel[Up, Down]] = intensity[1]
    workflow[ReducedSampleDataBySpinChannel[Down, Up]] = intensity[2]
    workflow[ReducedSampleDataBySpinChannel[Down, Down]] = intensity[3]
    workflow[FlipperEfficiency[Analyzer]] = FlipperEfficiency(f)

    result = np.zeros(4)
    for pola in [Up, Down]:
        for ana in [Up, Down]:
            contrib = workflow.compute(PolarizationCorrectedData[pola, ana])
            contrib = sc.concat(
                [contrib.upup, contrib.updown, contrib.downup, contrib.downdown],
                'dummy',
            )
            result += contrib.values
    np.testing.assert_allclose(result, ground_truth)


@pytest.mark.parametrize('f', [0.1, 0.5, 0.9, 0.99, 1.0])
def test_workflow_with_polarizer_flipper_computes_and_applies_matrix_inverse(
    f: float,
) -> None:
    ground_truth = np.array([7.0, 11.0, 13.0, 17.0])
    analyzer = np.array([[1.3, 0.9], [0.9, 1.3]])
    polarizer = np.array([[1.1, 0.7], [0.7, 1.1]])
    identity = np.array([[1.0, 0.0], [0.0, 1.0]])

    flipper = np.array([[1.0, 0.0], [1 - f, f]])
    intensity = (
        np.kron(identity, analyzer)
        @ np.kron(flipper @ polarizer, identity)
        @ ground_truth
    )

    workflow = CorrectionWorkflow()
    workflow[TransmissionFunction[Analyzer]] = FakeTransmissionFunction(analyzer)
    workflow[TransmissionFunction[Polarizer]] = FakeTransmissionFunction(polarizer)
    workflow[ReducedSampleDataBySpinChannel[Up, Up]] = intensity[0]
    workflow[ReducedSampleDataBySpinChannel[Up, Down]] = intensity[1]
    workflow[ReducedSampleDataBySpinChannel[Down, Up]] = intensity[2]
    workflow[ReducedSampleDataBySpinChannel[Down, Down]] = intensity[3]
    workflow[FlipperEfficiency[Polarizer]] = FlipperEfficiency(f)

    result = np.zeros(4)
    for pola in [Up, Down]:
        for ana in [Up, Down]:
            contrib = workflow.compute(PolarizationCorrectedData[pola, ana])
            contrib = sc.concat(
                [contrib.upup, contrib.updown, contrib.downup, contrib.downdown],
                'dummy',
            )
            result += contrib.values
    np.testing.assert_allclose(result, ground_truth)


@pytest.mark.parametrize('f1', [0.1, 0.5, 0.9, 0.99, 1.0])
@pytest.mark.parametrize('f2', [0.1, 0.5, 0.9, 0.99, 1.0])
def test_workflow_with_two_flipper_computes_and_applies_matrix_inverse(
    f1: float, f2: float
) -> None:
    ground_truth = np.array([7.0, 11.0, 13.0, 17.0])
    analyzer = np.array([[1.3, 0.9], [0.9, 1.3]])
    polarizer = np.array([[1.1, 0.7], [0.7, 1.1]])
    identity = np.array([[1.0, 0.0], [0.0, 1.0]])

    flipper1 = np.array([[1.0, 0.0], [1 - f1, f1]])
    flipper2 = np.array([[1.0, 0.0], [1 - f2, f2]])
    intensity = (
        np.kron(identity, analyzer @ flipper2)
        @ np.kron(flipper1 @ polarizer, identity)
        @ ground_truth
    )

    workflow = CorrectionWorkflow()
    workflow[TransmissionFunction[Analyzer]] = FakeTransmissionFunction(analyzer)
    workflow[TransmissionFunction[Polarizer]] = FakeTransmissionFunction(polarizer)
    workflow[ReducedSampleDataBySpinChannel[Up, Up]] = intensity[0]
    workflow[ReducedSampleDataBySpinChannel[Up, Down]] = intensity[1]
    workflow[ReducedSampleDataBySpinChannel[Down, Up]] = intensity[2]
    workflow[ReducedSampleDataBySpinChannel[Down, Down]] = intensity[3]
    workflow[FlipperEfficiency[Polarizer]] = FlipperEfficiency(f1)
    workflow[FlipperEfficiency[Analyzer]] = FlipperEfficiency(f2)

    result = np.zeros(4)
    for pola in [Up, Down]:
        for ana in [Up, Down]:
            contrib = workflow.compute(PolarizationCorrectedData[pola, ana])
            contrib = sc.concat(
                [contrib.upup, contrib.updown, contrib.downup, contrib.downdown],
                'dummy',
            )
            result += contrib.values
    np.testing.assert_allclose(result, ground_truth)


_he3_workflow = He3CellWorkflow()
_supermirror_workflow = SupermirrorWorkflow()
_polarizing_element_workflows = [_he3_workflow, _supermirror_workflow]


@pytest.mark.parametrize("polarizer_workflow", _polarizing_element_workflows)
@pytest.mark.parametrize("analyzer_workflow", _polarizing_element_workflows)
def test_polarization_analysis_workflow_creation(
    polarizer_workflow: sciline.Pipeline, analyzer_workflow: sciline.Pipeline
) -> None:
    workflow = PolarizationAnalysisWorkflow(
        polarizer_workflow=polarizer_workflow, analyzer_workflow=analyzer_workflow
    )
    handler = sciline.HandleAsComputeTimeException()
    for elem, wf in zip(
        (Polarizer, Analyzer), (polarizer_workflow, analyzer_workflow), strict=True
    ):
        graph = workflow.get(TransmissionFunction[elem], handler=handler)
        assert (
            pol.supermirror.SupermirrorEfficiencyFunction[elem] in graph.keys()
        ) == (wf is _supermirror_workflow)
        assert (pol.he3.He3CellPressure[elem] in graph.keys()) == (wf is _he3_workflow)


@pytest.mark.parametrize('f', [0.1, 0.5, 0.9, 0.99, 1.0])
def test_half_polarized_with_flipper_computes_and_applies_matrix_inverse(
    f: float,
) -> None:
    ground_truth = np.array([7.0, 11.0])
    polarizer = np.array([[0.96, 0.04], [0.04, 0.96]])
    flipper = np.array([[1.0, 0.0], [1 - f, f]])
    intensity = flipper @ polarizer @ ground_truth

    workflow = CorrectionWorkflow(half_polarized=True)
    workflow[TransmissionFunction[Polarizer]] = FakeTransmissionFunction(polarizer)
    workflow[ReducedSampleDataBySpinChannel[Up, pol.NoAnalyzer]] = intensity[0]
    workflow[ReducedSampleDataBySpinChannel[Down, pol.NoAnalyzer]] = intensity[1]
    workflow[FlipperEfficiency[Polarizer]] = FlipperEfficiency(f)

    result = np.zeros(2)
    for pola in [Up, Down]:
        contrib = workflow.compute(HalfPolarizedCorrectedData[pola])
        contrib = sc.concat([contrib.up, contrib.down], 'dummy')
        result += contrib.values
    np.testing.assert_allclose(result, ground_truth)
