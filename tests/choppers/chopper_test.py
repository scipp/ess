# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc

import ess.choppers as ch


@pytest.fixture
def params():
    dim = 'frame'
    return {
        'frequency': sc.scalar(56.0, unit="Hz"),
        'phase': sc.scalar(0.5, unit='rad'),
        'position': sc.vector(value=[0.0, 0.0, 5.0], unit='m'),
        'cutout_angles_center': sc.linspace(
            dim=dim, start=0.25, stop=2.0 * np.pi, num=6, unit='rad'
        ),
        'cutout_angles_width': sc.linspace(
            dim=dim, start=0.1, stop=0.6, num=6, unit='rad'
        ),
        'kind': sc.scalar('wfm'),
    }


def test_make_chopper_bad_widths(params):
    params['cutout_angles_width'].values[1] = -3.0
    with pytest.raises(ValueError) as e_info:
        _ = ch.make_chopper(**params)
    assert str(e_info.value) == "Negative window width found in chopper cutout angles."


def test_make_chopper_bad_centers(params):
    params['cutout_angles_center'].values = params['cutout_angles_center'].values[
        [1, 0, 2, 3, 4, 5]
    ]
    with pytest.raises(ValueError) as e_info:
        _ = ch.make_chopper(**params)
    assert str(e_info.value) == "Chopper begin cutout angles are not monotonic."


def test_make_chopper_bad_begin_angles(params):
    cutout_angles_begin = (
        params['cutout_angles_center'] - 0.5 * params['cutout_angles_width']
    )
    cutout_angles_end = (
        params['cutout_angles_center'] + 0.5 * params['cutout_angles_width']
    )
    cutout_angles_begin.values = cutout_angles_begin.values[[1, 0, 2, 3, 4, 5]]
    with pytest.raises(ValueError) as e_info:
        _ = ch.make_chopper(
            frequency=params['frequency'],
            phase=params['phase'],
            position=params['position'],
            cutout_angles_begin=cutout_angles_begin,
            cutout_angles_end=cutout_angles_end,
            kind=params['kind'],
        )
    # This will raise the error on the widths before it reaches the monotonicity check
    assert str(e_info.value) == "Negative window width found in chopper cutout angles."


def test_make_chopper_bad_close_angles(params):
    dim = 'frame'
    with pytest.raises(ValueError) as e_info:
        _ = ch.make_chopper(
            frequency=params['frequency'],
            phase=params['phase'],
            position=params['position'],
            cutout_angles_begin=sc.array(
                dims=[dim], values=[0.0, 1.0, 2.0], unit='rad'
            ),
            cutout_angles_end=sc.array(dims=[dim], values=[4.0, 3.0, 5.0], unit='rad'),
            kind=params['kind'],
        )
    assert str(e_info.value) == "Chopper end cutout angles are not monotonic."


def test_angular_frequency(params):
    chopper = ch.make_chopper(**params)
    assert sc.identical(
        ch.angular_frequency(chopper),
        (2.0 * np.pi * sc.units.rad) * params['frequency'],
    )


def test_cutout_angles_from_centers_widths(params):
    chopper = ch.make_chopper(**params)
    assert sc.allclose(
        ch.cutout_angles_begin(chopper),
        params["cutout_angles_center"] - 0.5 * params["cutout_angles_width"],
    )
    assert sc.allclose(
        ch.cutout_angles_end(chopper),
        params["cutout_angles_center"] + 0.5 * params["cutout_angles_width"],
    )


def test_cutout_angles_from_begin_end(params):
    dim = 'frame'
    del params['cutout_angles_center']
    del params['cutout_angles_width']
    params["cutout_angles_begin"] = sc.linspace(
        dim=dim, start=0.0, stop=1.5 * np.pi, num=6, unit='rad'
    )
    params["cutout_angles_end"] = sc.linspace(
        dim=dim, start=0.1, stop=2.0 * np.pi, num=6, unit='rad'
    )
    chopper = ch.make_chopper(**params)
    assert sc.allclose(
        ch.cutout_angles_width(chopper),
        params["cutout_angles_end"] - params["cutout_angles_begin"],
    )
    assert sc.allclose(
        ch.cutout_angles_center(chopper),
        0.5 * (params["cutout_angles_begin"] + params["cutout_angles_end"]),
    )


def test_time_open_closed(params):
    dim = 'frame'
    chopper = ch.make_chopper(
        frequency=sc.scalar(0.5, unit=sc.units.one / sc.units.s),
        phase=sc.scalar(0.0, unit='rad'),
        position=params['position'],
        cutout_angles_begin=sc.array(
            dims=[dim], values=np.pi * np.array([0.0, 0.5, 1.0]), unit='rad'
        ),
        cutout_angles_end=sc.array(
            dims=[dim], values=np.pi * np.array([0.5, 1.0, 1.5]), unit='rad'
        ),
        kind=params['kind'],
    )

    assert sc.allclose(
        ch.time_open(chopper),
        sc.to_unit(sc.array(dims=[dim], values=[0.0, 0.5, 1.0], unit='s'), 'us'),
    )
    assert sc.allclose(
        ch.time_closed(chopper),
        sc.to_unit(sc.array(dims=[dim], values=[0.5, 1.0, 1.5], unit='s'), 'us'),
    )

    chopper["phase"] = sc.scalar(2.0 * np.pi / 3.0, unit='rad')
    assert sc.allclose(
        ch.time_open(chopper),
        sc.to_unit(
            sc.array(
                dims=[dim], values=np.array([0.0, 0.5, 1.0]) + 2.0 / 3.0, unit='s'
            ),
            'us',
        ),
    )
    assert sc.allclose(
        ch.time_closed(chopper),
        sc.to_unit(
            sc.array(
                dims=[dim], values=np.array([0.5, 1.0, 1.5]) + 2.0 / 3.0, unit='s'
            ),
            'us',
        ),
    )


def test_find_chopper_keys():
    da = sc.DataArray(
        data=sc.scalar('dummy'),
        coords={
            'chopper3': sc.scalar(0),
            'abc': sc.scalar(0),
            'chopper_1': sc.scalar(0),
            'sample': sc.scalar(0),
            'source': sc.scalar(0),
            'Chopper_wfm': sc.scalar(0),
            'chopper0': sc.scalar(0),
            'chopper5': sc.scalar(0),
            'monitor': sc.scalar(0),
        },
    )
    expected = ['chopper3', 'chopper_1', 'Chopper_wfm', 'chopper0', 'chopper5']
    assert ch.find_chopper_keys(da) == expected
