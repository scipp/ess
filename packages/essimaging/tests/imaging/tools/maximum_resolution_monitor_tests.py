import numpy as np
import pytest
import scipp as sc

from ess.imaging.tools import maximum_resolution_achievable


def test_finds_maximum_resolution():
    events = sc.DataArray(
        sc.ones(dims=['events'], shape=(100,)),
        coords={
            'x': sc.array(
                dims=['events'],
                values=np.tile(np.linspace(1 / 10 / 2, 1 - 1 / 10 / 2, 10), 10),
            ),
            'y': sc.array(
                dims=['events'],
                values=np.repeat(np.linspace(1 / 10 / 2, 1 - 1 / 10 / 2, 10), 10),
            ),
            't': sc.ones(dims=['events'], shape=(100,)),
        },
    )

    x_be, y_be = maximum_resolution_achievable(
        events,
        sc.linspace('x', 0, 1, 2),
        sc.linspace('x', 0, 1, 2),
        sc.linspace('t', 0, 2, 2),
    )
    assert len(x_be) == 11
    assert len(y_be) == 11
    assert x_be[0] == 0
    assert x_be[-1] == 1
    assert y_be[0] == 0
    assert y_be[-1] == 1


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_finds_maximum_resolution_random(seed):
    np.random.seed(seed)
    n = np.random.randint(1000, 100_000)
    events = sc.DataArray(
        sc.ones(dims=['events'], shape=(n,)),
        coords={
            'x': sc.array(dims=['events'], values=np.random.random(n)),
            'y': sc.array(dims=['events'], values=np.random.random(n)),
            't': sc.ones(dims=['events'], shape=(n,)),
        },
    )
    x_be, y_be = maximum_resolution_achievable(
        events,
        sc.linspace('x', 0, 1, 2),
        sc.linspace('y', 0, 1, 2),
        sc.linspace('t', 0, 2, 2),
        # Need enough tries to be sure we find the optimum
        max_tries=100,
    )
    assert (
        events.bin(x=x_be, y=y_be, t=sc.linspace('t', 0, 2, 2)).bins.size().min().value
        > 0
    )
    assert (
        events.bin(
            x=sc.linspace('x', 0, 1, len(x_be) + 1),
            y=sc.linspace('y', 0, 1, len(y_be) + 1),
            t=sc.linspace('t', 0, 2, 2),
        )
        .bins.size()
        .min()
        .value
        == 0
    )
