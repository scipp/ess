# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from dataclasses import replace

import pytest
import scipp as sc
import scippnexus as snx
from ess.beer import (
    BeerModulationAutoMcStasWorkflow,
    BeerModulationKnownPeaksMcStasWorkflow,
    BeerPowderMcStasWorkflow,
)
from ess.beer.mcstas.beamline import (
    ModulationMode,
    PulseShapingMode,
    simulation_choppers,
)
from ess.beer.parameters import beamline_parameters
from ess.beer.types import BeamlineParameters
from ess.powder.types import EmptyDetector, SampleRun, VanadiumRun
from scipp.testing import assert_allclose, assert_identical

from ess.reduce.nexus.types import DiskChoppers, Position


@pytest.mark.parametrize(
    ('mode', 'chopper_distance'),
    [
        (PulseShapingMode.ps1, (6.450 + 7.375) / 2),
        (ModulationMode.m2, 9.300),
        (ModulationMode.m4, (9.300 + 9.350) / 2),
    ],
)
def test_beamline_parameters_use_chopper_origin_and_unweighted_pixel_means(
    mode, chopper_distance
):
    source = sc.vector([3.0, 4.0, 5.0], unit='m')
    sample = sc.vector([3.0, 4.0, 165.0], unit='m')
    choppers = simulation_choppers(mode, source)
    ltotal = sc.array(dims=['pixel'], values=[162.0, 164.0, float('nan')], unit='m')
    two_theta = sc.array(dims=['pixel'], values=[60.0, 120.0, float('nan')], unit='deg')

    result = beamline_parameters(
        sc.array(dims=['bound'], values=[1.0, 3.0], unit='angstrom'),
        ltotal,
        two_theta,
        source,
        sample,
        choppers,
    )

    assert_identical(result['lambda_min'], sc.scalar(1.0, unit='angstrom'))
    assert_identical(result['lambda_max'], sc.scalar(3.0, unit='angstrom'))
    assert_identical(result['lambda_0'], sc.scalar(2.0, unit='angstrom'))
    assert_allclose(result['L1'], sc.scalar(160.0 - chopper_distance, unit='m'))
    assert_allclose(
        result['L2'],
        sc.array(dims=['pixel'], values=[2.0, 4.0, float('nan')], unit='m'),
        equal_nan=True,
    )
    assert_allclose(result['Ltotal'], result['L1'] + result['L2'], equal_nan=True)
    assert_identical(result['two_theta'], two_theta)
    assert_identical(result['mean_L2'], sc.scalar(3.0, unit='m'))
    assert_allclose(
        result['mean_Ltotal'], sc.scalar(163.0 - chopper_distance, unit='m')
    )
    assert_identical(result['mean_two_theta'], sc.scalar(90.0, unit='deg'))
    for name, chopper in choppers.items():
        assert_identical(result['chopper_frequency'][name], chopper.frequency)


@pytest.mark.parametrize(
    ('factory', 'run_type', 'mode'),
    [
        (BeerPowderMcStasWorkflow, SampleRun, PulseShapingMode.ps1),
        (BeerPowderMcStasWorkflow, VanadiumRun, PulseShapingMode.ps1),
        (BeerModulationAutoMcStasWorkflow, SampleRun, ModulationMode.m2),
        (BeerModulationKnownPeaksMcStasWorkflow, SampleRun, ModulationMode.m4),
    ],
)
def test_workflows_compute_beamline_parameters_from_geometry_and_choppers(
    factory, run_type, mode
):
    workflow = factory()
    source = sc.vector([0.0, 0.0, 0.0], unit='m')
    workflow[Position[snx.NXsource, run_type]] = source
    workflow[Position[snx.NXsample, run_type]] = sc.vector([0.0, 0.0, 160.0], unit='m')
    choppers = dict(simulation_choppers(mode, source))
    workflow[DiskChoppers[run_type]] = choppers
    workflow[EmptyDetector[run_type]] = sc.DataArray(
        sc.ones(dims=['pixel'], shape=[2]),
        coords={
            'position': sc.vectors(
                dims=['pixel'], values=[[2.0, 0.0, 160.0], [4.0, 0.0, 160.0]], unit='m'
            )
        },
    )

    # No filename, events, monitor data, or known peak positions are supplied.
    result = workflow.compute(BeamlineParameters[run_type])

    assert isinstance(result, sc.DataGroup)
    assert result['lambda_min'] < result['lambda_0'] < result['lambda_max']
    assert_allclose(result['L1'] + result['L2'], result['Ltotal'])
    assert_identical(result['mean_L2'], sc.scalar(3.0, unit='m'))
    assert_allclose(
        result['mean_two_theta'].to(unit='deg'), sc.scalar(90.0, unit='deg')
    )

    # A parked chopper must affect neither the effective origin nor the frequencies.
    choppers['PSC_parked'] = replace(
        next(iter(choppers.values())),
        frequency=sc.scalar(0.0, unit='Hz'),
        axle_position=sc.vector([0.0, 0.0, 20.0], unit='m'),
    )
    workflow[DiskChoppers[run_type]] = choppers
    assert_identical(workflow.compute(BeamlineParameters[run_type]), result)
