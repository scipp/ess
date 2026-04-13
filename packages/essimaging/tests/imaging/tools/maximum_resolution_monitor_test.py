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

    rebinned = maximum_resolution_achievable(
        events.bin(x=1024, y=1024),
        ('x', 'y'),
    )
    assert rebinned.sizes['x'] == 8
    assert rebinned.sizes['y'] == 8


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
    events = events.bin(x=2**10, y=2**10)
    del events.bins.coords['x']
    del events.bins.coords['y']
    rebinned = maximum_resolution_achievable(
        events,
        ('x', 'y'),
    )

    assert rebinned.bins.size().min().value > 0
    assert (
        events.fold('x', sizes={'x': 2 * rebinned.sizes['x'], '_x_aux': -1})
        .fold('y', sizes={'y': 2 * rebinned.sizes['y'], '_y_aux': -1})
        .bins.concat(['_x_aux', '_y_aux'])
        .bins.size()
        .min()
        .value
    ) == 0


def test_finds_maximum_resolution_time_binned_input():
    np.random.seed(0)
    n = 100_000
    events = sc.DataArray(
        sc.ones(dims=['events'], shape=(n,)),
        coords={
            'x': sc.array(dims=['events'], values=np.random.random(n)),
            'y': sc.array(dims=['events'], values=np.random.random(n)),
            't': sc.array(dims=['events'], values=np.random.random(n)),
        },
    )
    events = events.bin(x=128, y=128, t=500)
    rebinned = maximum_resolution_achievable(events, ('x', 'y'))

    assert rebinned.bins.size().min().value > 0
    assert (
        events.fold('x', sizes={'x': 2 * rebinned.sizes['x'], '_x_aux': -1})
        .fold('y', sizes={'y': 2 * rebinned.sizes['y'], '_y_aux': -1})
        .bins.concat(['_x_aux', '_y_aux'])
        .bins.size()
        .min()
        .value
    ) == 0


def test_finds_maximum_resolution_non_even_number_of_bins():
    np.random.seed(0)
    n = 100_000
    events = sc.DataArray(
        sc.ones(dims=['events'], shape=(n,)),
        coords={
            'x': sc.array(dims=['events'], values=np.random.random(n)),
            'y': sc.array(dims=['events'], values=np.random.random(n)),
            't': sc.array(dims=['events'], values=np.random.random(n)),
        },
    )
    events = events.bin(x=3**4 * 5, y=3**3 * 5, t=101)
    rebinned = maximum_resolution_achievable(events, ('x', 'y'))

    assert rebinned.bins.size().min().value > 0
    assert (
        events.fold('x', sizes={'x': 45, '_x_aux': -1})
        .fold('y', sizes={'y': 15, '_y_aux': -1})
        .bins.concat(['_x_aux', '_y_aux'])
        .bins.size()
        .min()
        .value
    ) == 0


def test_raises_if_not_binned():
    np.random.seed(0)
    n = 100_000
    events = sc.DataArray(
        sc.ones(dims=['events'], shape=(n,)),
        coords={
            'x': sc.array(dims=['events'], values=np.random.random(n)),
            'y': sc.array(dims=['events'], values=np.random.random(n)),
            't': sc.array(dims=['events'], values=np.random.random(n)),
        },
    )
    with pytest.raises(ValueError, match='Input data must be binned'):
        maximum_resolution_achievable(events, ('x', 'y'))


def test_raises_if_reaches_coarsest_grid_without_success():
    np.random.seed(0)
    n = 100
    events = sc.DataArray(
        sc.ones(dims=['events'], shape=(n,)),
        coords={
            'x': sc.array(dims=['events'], values=np.random.random(n)),
            'y': sc.array(dims=['events'], values=np.random.random(n)),
            't': sc.array(dims=['events'], values=np.random.random(n)),
        },
    )
    events = events.bin(x=128, y=128, t=101)
    with pytest.raises(ValueError, match='Even at the coarsest'):
        maximum_resolution_achievable(events, ('x', 'y'))


def test_one_dimension_denser_than_the_other():
    np.random.seed(0)
    n = 100
    events = sc.DataArray(
        sc.ones(dims=['events'], shape=(n,)),
        coords={
            'x': sc.array(dims=['events'], values=np.random.random(n)),
            'y': sc.array(dims=['events'], values=np.random.random(n)),
            't': sc.array(dims=['events'], values=np.random.random(n)),
        },
    )
    events = events.bin(x=2, y=128, t=2)
    rebinned = maximum_resolution_achievable(events, ('x', 'y'))
    assert rebinned.bins.size().min().value > 0
    assert rebinned.sizes['y'] > 4
