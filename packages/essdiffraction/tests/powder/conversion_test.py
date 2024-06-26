# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
import scipp.testing
import scippneutron as scn

from ess.powder.conversion import (
    add_scattering_coordinates_from_positions,
    to_dspacing_with_calibration,
)


@pytest.fixture(params=['random', 'zero'])
def calibration(request):
    rng = np.random.default_rng(789236)
    n = 30
    ds = sc.Dataset(
        data={
            'difa': sc.array(
                dims=['spectrum'],
                values=rng.uniform(1.0e2, 1.0e3, n),
                unit='us / angstrom**2',
            ),
            'difc': sc.array(
                dims=['spectrum'],
                values=rng.uniform(1.0e3, 1.0e4, n),
                unit='us / angstrom',
            ),
            'tzero': sc.array(
                dims=['spectrum'], values=rng.uniform(-1e2, 1e2, n), unit='us'
            ),
            'mask': sc.full(dims=['spectrum'], shape=[n], value=False, unit=None),
        },
        coords={'spectrum': sc.arange('spectrum', n, unit=None)},
    )
    if request.param == 'zero':
        ds['difa'].data = sc.zeros_like(ds['difa'].data)
    return ds


def test_dspacing_with_calibration_roundtrip(calibration):
    initial_tof = sc.DataArray(
        sc.ones(dims=['spectrum', 'tof'], shape=[calibration.sizes['spectrum'], 27]),
        coords={
            'spectrum': calibration.coords['spectrum'],
            'tof': sc.linspace('tof', 1.0, 1000.0, 27, unit='us'),
        },
    )
    dspacing = to_dspacing_with_calibration(initial_tof, calibration=calibration)

    d = dspacing.coords['dspacing']
    difa = calibration['difa'].data
    difc = calibration['difc'].data
    tzero = calibration['tzero'].data
    recomputed_tof = difa * d**2 + difc * d + tzero
    recomputed_tof = recomputed_tof.rename_dims({'dspacing': 'tof'})
    assert sc.allclose(recomputed_tof, initial_tof.coords['tof'])


def test_dspacing_with_calibration_roundtrip_with_wavelength(calibration):
    initial_wavelength = sc.DataArray(
        sc.ones(
            dims=['spectrum', 'wavelength'], shape=[calibration.sizes['spectrum'], 27]
        ),
        coords={
            'spectrum': calibration.coords['spectrum'],
            'wavelength': sc.linspace('wavelength', 10.0, 100.0, 27, unit='angstrom'),
            'tof': sc.linspace('wavelength', 1.0, 1000.0, 27, unit='us'),
        },
    )
    dspacing = to_dspacing_with_calibration(initial_wavelength, calibration=calibration)

    d = dspacing.coords['dspacing']
    difa = calibration['difa'].data
    difc = calibration['difc'].data
    tzero = calibration['tzero'].data
    recomputed_tof = difa * d**2 + difc * d + tzero
    recomputed_tof = recomputed_tof.rename_dims({'dspacing': 'tof'})
    assert sc.allclose(
        recomputed_tof,
        initial_wavelength.coords['tof'].rename_dims({'wavelength': 'tof'}),
    )


def test_dspacing_with_calibration_consumes_positions(calibration):
    rng = np.random.default_rng(9274)
    n_spectra = calibration.sizes['spectrum']
    tof = sc.DataArray(
        sc.ones(dims=['spectrum', 'tof'], shape=[calibration.sizes['spectrum'], 27]),
        coords={
            'spectrum': calibration.coords['spectrum'],
            'tof': sc.linspace('tof', 1.0, 1000.0, 27, unit='us'),
            'position': sc.vectors(
                dims=['spectrum'],
                values=rng.uniform(-2.0, 2.0, (n_spectra, 3)),
                unit='m',
            ),
            'sample_position': sc.vector(value=[0.1, 0.02, 0.0], unit='m'),
            'source_position': sc.vector(value=[-10.0, -1.0, 0.0], unit='m'),
        },
    )
    dspacing = to_dspacing_with_calibration(tof, calibration=calibration)
    assert sc.identical(dspacing.coords['position'], tof.coords['position'])
    assert not dspacing.coords['position'].aligned
    assert sc.identical(
        dspacing.coords['sample_position'], tof.coords['sample_position']
    )
    assert not dspacing.coords['sample_position'].aligned
    assert sc.identical(
        dspacing.coords['source_position'], tof.coords['source_position']
    )
    assert not dspacing.coords['source_position'].aligned


def test_dspacing_with_calibration_does_not_use_positions(calibration):
    rng = np.random.default_rng(91032)
    n_spectra = calibration.sizes['spectrum']
    tof_no_pos = sc.DataArray(
        sc.ones(dims=['spectrum', 'tof'], shape=[n_spectra, 27]),
        coords={
            'spectrum': calibration.coords['spectrum'],
            'tof': sc.linspace('tof', 1.0, 1000.0, 27, unit='us'),
        },
    )
    tof_pos = tof_no_pos.copy()
    tof_pos.coords['position'] = sc.vectors(
        dims=['spectrum'], values=rng.uniform(-2.0, 2.0, (n_spectra, 3)), unit='m'
    )
    tof_pos.coords['sample_position'] = sc.vector(value=[0.1, 0.02, 0.0], unit='m')
    tof_pos.coords['source_position'] = sc.vector(value=[-10.0, -1.0, 0.0], unit='m')

    dspacing_no_pos = to_dspacing_with_calibration(tof_no_pos, calibration=calibration)
    dspacing_pos = to_dspacing_with_calibration(tof_pos, calibration=calibration)

    assert sc.allclose(
        dspacing_no_pos.coords['dspacing'], dspacing_pos.coords['dspacing']
    )


def test_add_scattering_coordinates_from_positions():
    position = sc.vectors(
        dims=['spectrum'], values=np.arange(14 * 3).reshape((14, 3)), unit='m'
    )
    sample_position = sc.vector([0.0, 0.0, 0.01], unit='m')
    source_position = sc.vector([0.0, 0.0, -11.3], unit='m')
    tof = sc.DataArray(
        sc.ones(dims=['spectrum', 'tof'], shape=[14, 27]),
        coords={
            'position': position,
            'tof': sc.linspace('tof', 1.0, 1000.0, 27, unit='us'),
            'sample_position': sample_position,
            'source_position': source_position,
        },
    )
    graph = {
        **scn.conversion.graph.beamline.beamline(scatter=True),
        **scn.conversion.graph.tof.elastic('tof'),
    }

    result = add_scattering_coordinates_from_positions(tof, graph)

    assert 'wavelength' in result.coords
    assert 'two_theta' in result.coords
