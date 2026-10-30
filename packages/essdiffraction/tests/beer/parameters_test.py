# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from dataclasses import replace

import pytest
import scipp as sc
from ess.beer.conversions import (
    effective_chopper_position,
    modulation_beamline_geometry_graph,
    powder_beamline_geometry_graph,
)
from ess.beer.parameters import beamline_parameters
from ess.beer.types import BeamlineGeometryGraph, SampleRun
from scipp.testing import assert_allclose, assert_identical
from scippneutron.chopper import DiskChopper


@pytest.fixture
def choppers():
    mca = DiskChopper(
        axle_position=sc.vector([0.0, 1.0, 6.0], unit='m'),
        frequency=sc.scalar(10.0, unit='Hz'),
        beam_position=sc.scalar(0.0, unit='deg'),
        phase=sc.scalar(0.0, unit='deg'),
        slit_begin=sc.array(dims=['slit'], values=[0.0], unit='deg'),
        slit_end=sc.array(dims=['slit'], values=[30.0], unit='deg'),
    )
    return {
        'MCA': mca,
        'MCB': replace(
            mca,
            axle_position=sc.vector([0.0, 2.0, 10.0], unit='m'),
            frequency=sc.scalar(-10.0, unit='Hz'),
        ),
        'FC1A': replace(
            mca,
            axle_position=sc.vector([0.0, 0.0, 100.0], unit='m'),
            frequency=sc.scalar(14.0, unit='Hz'),
        ),
    }


def test_beamline_parameters_use_supplied_graph_and_nanmeans(choppers):
    geometry = {
        'L1': sc.scalar(100.0, unit='m'),
        'L2': sc.array(dims=['pixel'], values=[2.0, 4.0, float('nan')], unit='m'),
        'Ltotal': sc.array(
            dims=['pixel'], values=[102.0, 104.0, float('nan')], unit='m'
        ),
        'two_theta': sc.array(
            dims=['pixel'], values=[60.0, 120.0, float('nan')], unit='deg'
        ),
    }
    detector = sc.DataArray(
        sc.ones(sizes={'pixel': 3}),
        coords={f'input_{name}': value for name, value in geometry.items()},
    )
    graph = BeamlineGeometryGraph[SampleRun](
        {name: f'input_{name}' for name in geometry}
    )

    result = beamline_parameters(
        sc.array(dims=['bound'], values=[1.0, 3.0], unit='angstrom'),
        detector,
        graph,
        choppers,
    )

    expected = sc.DataGroup(
        **geometry,
        lambda_min=sc.scalar(1.0, unit='angstrom'),
        lambda_max=sc.scalar(3.0, unit='angstrom'),
        lambda_0=sc.scalar(2.0, unit='angstrom'),
        mean_L2=sc.scalar(3.0, unit='m'),
        mean_Ltotal=sc.scalar(103.0, unit='m'),
        mean_two_theta=sc.scalar(90.0, unit='deg'),
        chopper_frequency=sc.DataGroup(
            MCA=sc.scalar(10.0, unit='Hz'),
            MCB=sc.scalar(-10.0, unit='Hz'),
            FC1A=sc.scalar(14.0, unit='Hz'),
        ),
    )
    assert_identical(result, expected)


@pytest.mark.parametrize(
    ('modulation', 'chopper_unit', 'expected_l1'),
    [(False, 'm', 160.0), (True, 'm', 152.0), (True, 'cm', 152.0)],
    ids=['powder', 'modulation', 'modulation-cm'],
)
def test_beamline_geometry(modulation, chopper_unit, expected_l1, choppers):
    source = sc.vector([3.0, 4.0, 5.0], unit='m')
    sample = source + sc.vector([0.0, 0.0, 160.0], unit='m')
    gravity = sc.vector([0.0, -9.81, 0.0], unit='m/s^2')
    choppers = {
        name: replace(
            ch, axle_position=(source + ch.axle_position).to(unit=chopper_unit)
        )
        for name, ch in choppers.items()
    }
    graph = (
        modulation_beamline_geometry_graph(source, sample, gravity, choppers)
        if modulation
        else powder_beamline_geometry_graph(source, sample, gravity)
    )
    detector = sc.DataArray(
        sc.scalar(0.0),
        coords={'position': sample + sc.vector([2.0, 0.0, 0.0], unit='m')},
    )

    coords = detector.transform_coords(
        ['L1', 'L2', 'Ltotal', 'two_theta'], graph=graph
    ).coords

    # Only the longitudinal positions of MCA and MCB set the modulation origin.
    assert_allclose(coords['L1'], sc.scalar(expected_l1, unit='m'))
    assert_allclose(coords['L2'], sc.scalar(2.0, unit='m'))
    assert_allclose(coords['Ltotal'], sc.scalar(expected_l1 + 2.0, unit='m'))
    assert_allclose(coords['two_theta'].to(unit='deg'), sc.scalar(90.0, unit='deg'))


def test_effective_chopper_position_rejects_renamed_chopper(choppers):
    choppers['renamed_MCB'] = choppers.pop('MCB')

    with pytest.raises(ValueError, match=r'Unrecognized BEER chopper names.*MCB'):
        effective_chopper_position(choppers)


def test_effective_chopper_position_requires_definition_choppers(choppers):
    with pytest.raises(
        ValueError, match='No BEER pulse-shaping or modulation choppers'
    ):
        effective_chopper_position({'FC1A': choppers['FC1A']})
