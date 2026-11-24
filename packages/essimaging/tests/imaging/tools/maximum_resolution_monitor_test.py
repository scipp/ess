import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.imaging.tools import maximum_resolution_achievable


def _events(counts, dims=('x', 'y'), event_dim='event'):
    """Construct tiny bins directly, without a multi-dimensional binning workspace."""
    counts = np.asarray(counts, dtype=np.int64)
    ends = counts.cumsum().reshape(counts.shape)
    data = sc.DataArray(
        sc.arange(event_dim, int(counts.sum()), dtype='float64', unit='counts'),
        coords={'event_id': sc.arange(event_dim, int(counts.sum()), unit=None)},
    )
    return sc.DataArray(
        sc.bins(
            data=data,
            dim=event_dim,
            begin=sc.array(dims=dims, values=ends - counts, unit=None),
            end=sc.array(dims=dims, values=ends, unit=None),
        )
    )


@pytest.mark.parametrize(
    ('shape', 'stride', 'expected_shape'),
    [
        ((4, 4), (1, 1), (4, 4)),
        ((8, 8), (2, 2), (4, 4)),
        ((8, 8), (4, 4), (2, 2)),
        ((15, 9), (3, 3), (5, 3)),
        ((12, 12), (3, 3), (4, 4)),
        ((2, 16), (2, 2), (1, 8)),
        ((2, 16), (2, 8), (1, 2)),
        ((1, 8), (1, 2), (1, 4)),
        ((7, 5), (7, 5), (1, 1)),
    ],
)
def test_finds_first_populated_grid(shape, stride, expected_shape):
    counts = np.zeros(shape, dtype=np.int64)
    counts[:: stride[0], :: stride[1]] = 1
    events = _events(counts)
    original = events.copy()

    result = maximum_resolution_achievable(events, ('x', 'y'))

    assert result.shape == expected_shape
    assert result.bins.size().min().value > 0
    assert result.bins.size().sum().value == counts.sum()
    np.testing.assert_array_equal(
        np.sort(result.bins.constituents['data'].coords['event_id'].values),
        np.arange(counts.sum()),
    )
    assert_identical(events, original)


@pytest.mark.parametrize('dim', ['wavelength', 'event_time_offset'])
def test_preserves_other_binning_and_metadata(dim):
    counts = np.ones((4, 4, 3), dtype=np.int64)
    counts[1::2, :, 1] = 0
    events = _events(counts, dims=('row', 'col', dim))
    events.coords[dim] = sc.arange(dim, 4.0, unit='us')
    events.coords['run'] = sc.scalar(42)
    events.coords['row'] = sc.arange('row', 5.0, unit='m')
    events.coords['position'] = sc.vectors(
        dims=['row', 'col'], values=np.zeros((4, 4, 3)), unit='m'
    )
    original = events.copy()

    result = maximum_resolution_achievable(events, ('row', 'col'))

    assert result.sizes == {'row': 2, 'col': 2, dim: 3}
    expected_counts = counts.reshape(2, 2, 2, 2, 3).sum(axis=(1, 3))
    np.testing.assert_array_equal(result.bins.size().values, expected_counts)
    assert_identical(result.coords[dim], events.coords[dim])
    assert_identical(result.coords['run'], events.coords['run'])
    assert 'row' not in result.coords
    assert 'position' not in result.coords
    assert_identical(events, original)


def test_handles_transposed_input_and_multiple_other_dimensions():
    counts = np.zeros((4, 4, 2, 3), dtype=np.int64)
    counts[::2, ::2, :, :] = 1
    events = _events(counts, dims=('x', 'y', 'wavelength', 'pulse'))
    events = events.transpose(['pulse', 'y', 'wavelength', 'x'])

    result = maximum_resolution_achievable(events, ('x', 'y'))

    assert result.sizes == {'pulse': 3, 'y': 2, 'wavelength': 2, 'x': 2}
    assert result.bins.size().min().value == 1
    assert result.bins.size().sum().value == counts.sum()


def test_counts_events_instead_of_event_weights():
    events = _events(np.ones((2, 2), dtype=np.int64))
    events.bins.data *= 0.0
    result = maximum_resolution_achievable(events, ('x', 'y'))
    assert result.sizes == {'x': 2, 'y': 2}
    assert result.bins.size().min().value == 1


def test_excludes_masked_pixels_from_counts_and_output():
    events = _events(np.ones((4, 4), dtype=np.int64))
    events.masks['bad_pixel'] = sc.array(dims=['x'], values=[True, False, True, False])
    original = events.copy()

    result = maximum_resolution_achievable(events, ('x', 'y'))

    assert result.sizes == {'x': 2, 'y': 2}
    assert result.bins.size().min().value == 2
    assert result.bins.size().sum().value == 8
    assert_identical(events, original)


@pytest.mark.parametrize('image_dims', [('x', 'y'), ('_x_aux', '_y_aux')])
def test_avoids_auxiliary_dimension_collisions(image_dims):
    counts = np.zeros((4, 4, 2), dtype=np.int64)
    counts[::2, ::2, :] = 1
    events = _events(counts, dims=(*image_dims, '_x_aux_'), event_dim='_y_aux_')
    result = maximum_resolution_achievable(events, image_dims)
    assert result.sizes == {image_dims[0]: 2, image_dims[1]: 2, '_x_aux_': 2}
    assert result.bins.size().min().value == 1


def test_search_concatenates_only_once_and_does_not_rebin(monkeypatch):
    counts = np.zeros((8, 8), dtype=np.int64)
    counts[0, 0] = 1
    events = _events(counts)
    concat = type(events.bins).concat
    calls = []

    def counted_concat(self, dim=None):
        calls.append(dim)
        return concat(self, dim)

    def forbidden_bin(*args, **kwargs):
        pytest.fail('The resolution search must not re-bin events.')

    monkeypatch.setattr(type(events.bins), 'concat', counted_concat)
    monkeypatch.setattr(sc.DataArray, 'bin', forbidden_bin)

    result = maximum_resolution_achievable(events, ('x', 'y'))

    assert result.sizes == {'x': 1, 'y': 1}
    assert len(calls) == 1


def test_raises_if_not_binned():
    with pytest.raises(ValueError, match='Input data must be binned'):
        maximum_resolution_achievable(
            sc.DataArray(sc.ones(sizes={'x': 2, 'y': 2})), ('x', 'y')
        )


@pytest.mark.parametrize('dims', [('x', 'x'), ('x',), ('x', 'y', 'z')])
def test_rejects_invalid_image_dimensions(dims):
    with pytest.raises(ValueError, match='two distinct dimensions'):
        maximum_resolution_achievable(_events([[1]]), dims)


def test_rejects_missing_image_dimension():
    with pytest.raises(ValueError, match='present and non-empty'):
        maximum_resolution_achievable(_events([[1]]), ('x', 'z'))


def test_rejects_empty_image_dimension():
    with pytest.raises(ValueError, match='present and non-empty'):
        maximum_resolution_achievable(_events(np.zeros((0, 2))), ('x', 'y'))


def test_rejects_empty_other_dimension():
    events = _events(np.zeros((2, 2, 0)), dims=('x', 'y', 'wavelength'))
    with pytest.raises(ValueError, match='at least one bin'):
        maximum_resolution_achievable(events, ('x', 'y'))


@pytest.mark.parametrize('empty', ['all', 'wavelength', 'masked'])
def test_raises_if_coarsest_grid_has_empty_bins(empty, monkeypatch):
    counts = np.ones((4, 4, 2), dtype=np.int64)
    if empty == 'all':
        counts[:] = 0
    elif empty == 'wavelength':
        counts[:, :, 1] = 0
    events = _events(counts, dims=('x', 'y', 'wavelength'))
    if empty == 'masked':
        events.masks['bad_pixel'] = sc.ones(sizes={'x': 4, 'y': 4}, dtype='bool')

    def forbidden_concat(*args, **kwargs):
        pytest.fail('An unsuccessful search must not concatenate events.')

    monkeypatch.setattr(type(events.bins), 'concat', forbidden_concat)
    with pytest.raises(ValueError, match='Even at the coarsest'):
        maximum_resolution_achievable(events, ('x', 'y'))
