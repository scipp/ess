# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Tests for how the beam center is applied, as opposed to how it is determined."""

import pytest
import scipp as sc
import scippnexus as snx
from ess.sans.common import gravity_vector
from ess.sans.conversions import sans_elastic
from ess.sans.types import BeamCenter, CorrectForGravity, Position, SampleRun

SAMPLE_POSITION = sc.vector([0.0, 0.0, 0.0], unit='m')
# Offset of the source from the sample, i.e., the nominal beam is along Z.
SOURCE_OFFSET = sc.vector([0.0, 0.0, -20.0], unit='m')
NO_BEAM_CENTER = sc.vector([0.0, 0.0, 0.0], unit='m')


def make_graph(
    *,
    beam_center: sc.Variable,
    correct_for_gravity: bool = False,
    sample_position: sc.Variable = SAMPLE_POSITION,
) -> dict:
    return sans_elastic(
        CorrectForGravity(correct_for_gravity),
        sample_position=Position[snx.NXsample, SampleRun](sample_position),
        source_position=Position[snx.NXsource, SampleRun](
            sample_position + SOURCE_OFFSET
        ),
        beam_center=BeamCenter(beam_center),
        gravity=gravity_vector(),
    )


def angle(position: sc.Variable, graph: dict, name: str = 'two_theta') -> sc.Variable:
    # A negligible wavelength keeps the gravity drop out of the comparison.
    da = sc.DataArray(
        data=sc.scalar(1.0),
        coords={'position': position, 'wavelength': sc.scalar(1e-6, unit='angstrom')},
    )
    return da.transform_coords(name, graph=graph).coords[name]


def assert_zero_angle(value: sc.Variable) -> None:
    assert sc.isclose(
        value, sc.scalar(0.0, unit='rad'), atol=sc.scalar(1e-9, unit='rad')
    )


def test_two_theta_is_zero_on_beam_axis_without_beam_center() -> None:
    position = sc.vector([0.0, 0.0, 5.0], unit='m')
    assert_zero_angle(angle(position, make_graph(beam_center=NO_BEAM_CENTER)))


def test_two_theta_is_zero_at_beam_center() -> None:
    beam_center = sc.vector([0.05, -0.03, 0.0], unit='m')
    position = sc.vector([0.05, -0.03, 5.0], unit='m')
    assert_zero_angle(angle(position, make_graph(beam_center=beam_center)))


def test_beam_center_shifts_two_theta() -> None:
    position = sc.vector([0.05, -0.03, 5.0], unit='m')
    without = angle(position, make_graph(beam_center=NO_BEAM_CENTER))
    with_center = angle(
        position, make_graph(beam_center=sc.vector([0.05, -0.03, 0.0], unit='m'))
    )
    assert without.value > 0.0
    assert with_center.value < without.value


def test_phi_is_measured_around_the_beam_center() -> None:
    beam_center = sc.vector([0.05, -0.03, 0.0], unit='m')
    # Directly to the right of the beam center, i.e., phi == 0.
    position = sc.vector([0.25, -0.03, 5.0], unit='m')
    assert_zero_angle(angle(position, make_graph(beam_center=beam_center), name='phi'))


@pytest.mark.parametrize('correct_for_gravity', [False, True])
def test_beam_center_is_applied_relative_to_sample_position(
    correct_for_gravity: bool,
) -> None:
    sample_position = sc.vector([0.1, -0.07, 0.0], unit='m')
    beam_center = sc.vector([0.05, -0.03, 0.0], unit='m')
    position = sample_position + beam_center + sc.vector([0, 0, 5.0], unit='m')
    graph = make_graph(
        beam_center=beam_center,
        correct_for_gravity=correct_for_gravity,
        sample_position=sample_position,
    )
    assert_zero_angle(angle(position, graph))
