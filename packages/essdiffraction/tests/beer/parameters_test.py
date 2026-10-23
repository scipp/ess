# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from dataclasses import replace

import scipp as sc
from ess.beer.parameters import beamline_parameters
from scipp.testing import assert_allclose, assert_identical
from scippneutron.chopper import DiskChopper


def test_beamline_parameters_use_chopper_origin_and_unweighted_pixel_means():
    source = sc.vector([3.0, 4.0, 5.0], unit='m')
    sample = sc.vector([3.0, 4.0, 165.0], unit='m')
    # Choppers at 6 m and 10 m from the source define an effective origin at 8 m.
    # Transverse axle offsets and the frame chopper must not affect this distance.
    psc1 = DiskChopper(
        axle_position=source + sc.vector([0.0, 1.0, 6.0], unit='m'),
        frequency=sc.scalar(10.0, unit='Hz'),
        beam_position=sc.scalar(0.0, unit='deg'),
        phase=sc.scalar(0.0, unit='deg'),
        slit_begin=sc.array(dims=['slit'], values=[0.0], unit='deg'),
        slit_end=sc.array(dims=['slit'], values=[30.0], unit='deg'),
    )
    choppers = {
        'PSC1': psc1,
        'PSC2': replace(
            psc1,
            axle_position=source + sc.vector([0.0, 2.0, 10.0], unit='m'),
            frequency=sc.scalar(-10.0, unit='Hz'),
        ),
        'FC1': replace(
            psc1,
            axle_position=source + sc.vector([0.0, 0.0, 100.0], unit='m'),
            frequency=sc.scalar(14.0, unit='Hz'),
        ),
    }
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
    assert_identical(result['L1'], sc.scalar(152.0, unit='m'))
    assert_allclose(
        result['L2'],
        sc.array(dims=['pixel'], values=[2.0, 4.0, float('nan')], unit='m'),
        equal_nan=True,
    )
    assert_allclose(
        result['Ltotal'],
        sc.array(dims=['pixel'], values=[154.0, 156.0, float('nan')], unit='m'),
        equal_nan=True,
    )
    assert_identical(result['two_theta'], two_theta)
    assert_identical(result['mean_L2'], sc.scalar(3.0, unit='m'))
    assert_identical(result['mean_Ltotal'], sc.scalar(155.0, unit='m'))
    assert_identical(result['mean_two_theta'], sc.scalar(90.0, unit='deg'))
    assert_identical(
        result['chopper_frequency'],
        sc.DataGroup(
            PSC1=sc.scalar(10.0, unit='Hz'),
            PSC2=sc.scalar(-10.0, unit='Hz'),
            FC1=sc.scalar(14.0, unit='Hz'),
        ),
    )
