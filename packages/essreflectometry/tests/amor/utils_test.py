# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# flake8: noqa: F403, F405

import pytest
import scipp as sc

from ess.amor.utils import q_theta_figure, wavelength_theta_figure, wavelength_z_figure


@pytest.fixture
def da():
    return sc.DataArray(
        data=sc.ones(dims=('events',), shape=(10,)),
        coords={
            'wavelength': sc.linspace('events', 1, 12, 10, unit='angstrom'),
            'theta': sc.linspace('events', 0, 1, 10, unit='deg'),
            'Q': sc.linspace('events', 0, 0.3, 10, unit='1/angstrom'),
            'z_index': sc.arange('events', 0, 10),
        },
    )


@pytest.fixture
def wavelength_bins():
    return sc.linspace('wavelength', 1, 12, 3, unit='angstrom')


@pytest.fixture
def theta_bins():
    return sc.linspace('theta', 0, 1, 3, unit='deg')


@pytest.fixture
def q_bins():
    return sc.linspace('Q', 0, 0.3, 3, unit='1/angstrom')


def test_wavelength_figure_table(da, wavelength_bins, theta_bins):
    with pytest.raises(ValueError, match='binning provided'):
        wavelength_theta_figure(da)

    with pytest.raises(ValueError, match='binning provided'):
        wavelength_theta_figure(da, wavelength_bins=wavelength_bins)

    with pytest.raises(ValueError, match='binning provided'):
        wavelength_theta_figure(da, theta_bins=theta_bins)

    assert wavelength_theta_figure(
        da, theta_bins=theta_bins, wavelength_bins=wavelength_bins
    )


def test_wavelength_figure_binned(da, wavelength_bins, theta_bins):
    with pytest.raises(ValueError, match='binning provided'):
        wavelength_theta_figure(da.bin(wavelength=3))

    with pytest.raises(ValueError, match='binning provided'):
        wavelength_theta_figure(da.bin(theta=3))

    assert wavelength_theta_figure(da.bin(wavelength=3, theta=3))
    assert wavelength_theta_figure(da.bin(wavelength=3), theta_bins=theta_bins)
    assert wavelength_theta_figure(da.bin(theta=3), wavelength_bins=wavelength_bins)


def test_wavelength_figure_hist(da, wavelength_bins, theta_bins):
    with pytest.raises(ValueError, match='must have wavelength and theta coord'):
        wavelength_theta_figure(da.hist(wavelength=3))

    with pytest.raises(ValueError, match='must have wavelength and theta coord'):
        wavelength_theta_figure(da.hist(theta=3))

    assert wavelength_theta_figure(da.hist(wavelength=3, theta=3))


def test_wavelength_figure_multiple_datasets(da, wavelength_bins, theta_bins):
    assert wavelength_theta_figure(
        (
            da.bin(wavelength=10, theta=10),
            da.hist(wavelength=10, theta=10),
        ),
    )
    assert wavelength_theta_figure(
        (
            da,
            da.bin(wavelength=10, theta=10),
            da.hist(wavelength=10, theta=10),
        ),
        wavelength_bins=wavelength_bins,
        theta_bins=theta_bins,
    )
    assert wavelength_theta_figure(
        (da, da),
        wavelength_bins=(wavelength_bins, wavelength_bins),
        theta_bins=theta_bins,
    )

    with pytest.raises(ValueError, match=r'zip\(\) argument .*'):
        assert wavelength_theta_figure(
            (da, da),
            # If sequence of bins is supplied, must have right length
            wavelength_bins=(wavelength_bins,),
            theta_bins=theta_bins,
        )

    assert wavelength_theta_figure(
        da,
        theta_bins=theta_bins,
        wavelength_bins=wavelength_bins,
        q_edges_to_display=(sc.scalar(1.0, unit='1/angstrom'),),
    )
    assert wavelength_theta_figure(
        da,
        theta_bins=theta_bins,
        wavelength_bins=wavelength_bins,
        # Can pass plot kwargs
        grid=True,
    )


def test_q_figure_table(da, q_bins, theta_bins):
    with pytest.raises(ValueError, match='binning provided'):
        q_theta_figure(da)

    with pytest.raises(ValueError, match='binning provided'):
        q_theta_figure(da, q_bins=q_bins)

    with pytest.raises(ValueError, match='binning provided'):
        q_theta_figure(da, theta_bins=theta_bins)

    assert q_theta_figure(da, theta_bins=theta_bins, q_bins=q_bins)


def test_q_figure_binned(da, q_bins, theta_bins):
    with pytest.raises(ValueError, match='binning provided'):
        q_theta_figure(da.bin(Q=3))

    with pytest.raises(ValueError, match='binning provided'):
        q_theta_figure(da.bin(theta=3))

    assert q_theta_figure(da.bin(Q=3, theta=3))
    assert q_theta_figure(da.bin(Q=3), theta_bins=theta_bins)
    assert q_theta_figure(da.bin(theta=3), q_bins=q_bins)


def test_q_figure_hist(da, q_bins, theta_bins):
    with pytest.raises(ValueError, match='must have theta and Q coord'):
        q_theta_figure(da.hist(Q=3))

    with pytest.raises(ValueError, match='must have theta and Q coord'):
        q_theta_figure(da.hist(theta=3))

    assert q_theta_figure(da.hist(Q=3, theta=3))


def test_q_figure_multiple_datasets(da, q_bins, theta_bins):
    assert q_theta_figure(
        (
            da.bin(Q=10, theta=10),
            da.hist(Q=10, theta=10),
        ),
    )
    assert q_theta_figure(
        (
            da,
            da.bin(Q=10, theta=10),
            da.hist(Q=10, theta=10),
        ),
        q_bins=q_bins,
        theta_bins=theta_bins,
    )
    assert q_theta_figure(
        (da, da),
        q_bins=(q_bins, q_bins),
        theta_bins=theta_bins,
    )

    with pytest.raises(ValueError, match=r'zip\(\) argument .*'):
        assert q_theta_figure(
            (da, da),
            # If sequence of bins is supplied, must have right length
            q_bins=(q_bins,),
            theta_bins=theta_bins,
        )

    assert q_theta_figure(
        da,
        theta_bins=theta_bins,
        q_bins=q_bins,
        # Can pass plot kwargs
        grid=True,
    )


def test_z_figure_binned(da, wavelength_bins):
    da = da.group('z_index').fold('z_index', dims=('blade', 'wire'), shape=(2, 5))

    with pytest.raises(ValueError, match='binning provided'):
        wavelength_z_figure(da)

    assert wavelength_z_figure(da.bin(wavelength=3))
    assert wavelength_z_figure(da, wavelength_bins=wavelength_bins)


def test_z_figure_hist(da, wavelength_bins):
    da = da.group('z_index').fold('z_index', dims=('blade', 'wire'), shape=(2, 5))
    assert wavelength_z_figure(da.hist(wavelength=3))


def test_z_figure_multiple_datasets(da, wavelength_bins):
    da = da.group('z_index').fold('z_index', dims=('blade', 'wire'), shape=(2, 5))
    assert wavelength_z_figure(
        (
            da.bin(wavelength=3),
            da.hist(wavelength=3),
        ),
    )
    assert wavelength_z_figure(
        (
            da,
            da.bin(wavelength=3),
            da.hist(wavelength=3),
        ),
        wavelength_bins=wavelength_bins,
    )
    assert wavelength_z_figure(
        (
            da.bin(wavelength=3),
            da.hist(wavelength=3),
        ),
        wavelength_bins=(wavelength_bins, wavelength_bins),
    )
    with pytest.raises(ValueError, match=r'zip\(\) argument .*'):
        assert wavelength_z_figure(
            (
                da.bin(wavelength=3),
                da.hist(wavelength=3),
            ),
            # If sequence of bins is supplied, must have right length
            wavelength_bins=(wavelength_bins,),
        )

    assert wavelength_z_figure(
        da,
        wavelength_bins=wavelength_bins,
        # Can pass plot kwargs
        grid=True,
    )
