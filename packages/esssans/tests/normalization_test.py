# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
import scippnexus as snx
from ess.isissans.data import sans2d_solid_angle_reference
from ess.sans import normalization

from ess.reduce.nexus.types import NeXusTransformation, SampleRun

# See https://github.com/mantidproject/mantid/blob/main/instrument/SANS2D_Definition_Tubes.xml  # noqa: E501
_SANS2D_PIXEL_RADIUS = 0.00405 * sc.Unit('m')
_SANS2D_PIXEL_LENGTH = 0.002033984375 * sc.Unit('m')


def _sans2d_geometry():
    R = _SANS2D_PIXEL_RADIUS.value
    L = _SANS2D_PIXEL_LENGTH.value
    pixel_shape = {
        'vertices': sc.vectors(
            dims=['vertex'],
            values=[
                [0, 0, 0],
                [R, 0, 0],
                [0, L, 0],
            ],
            unit='m',
        )
    }
    # Rotate +y to -x
    transform = NeXusTransformation[snx.NXdetector, SampleRun](
        sc.spatial.rotation(value=[0, 0, 1 / 2**0.5, 1 / 2**0.5])
    )
    return {'pixel_shape': pixel_shape, 'transform': transform}


def _mantid_sans2d_solid_angle_data():
    simpleapi = pytest.importorskip("mantid.simpleapi")
    scippneutron = pytest.importorskip("scippneutron")

    ws = simpleapi.Load('SANS2D00063091.nxs')
    radius = _SANS2D_PIXEL_RADIUS
    length = _SANS2D_PIXEL_LENGTH

    simpleapi.SetInstrumentParameter(
        ws,
        ParameterName='x-pixel-size',
        ParameterType='Number',
        Value=str((2 * radius).to(unit='mm').value),
    )
    simpleapi.SetInstrumentParameter(
        ws,
        ParameterName='y-pixel-size',
        ParameterType='Number',
        Value=str(length.to(unit='mm').value),
    )
    outWs = simpleapi.SolidAngle(ws, method='HorizontalTube')
    da = scippneutron.from_mantid(outWs)['data']['spectrum', :120000:100]
    # Create new reference file:
    # sc.io.hdf5.save_hdf5(da, 'SANS2D00063091.SolidAngle_from_mantid.h5')
    # Note, also update the registry version, don't overwrite old reference files.
    # Overwriting reference files breaks old versions of the repository.
    return da


@pytest.mark.filterwarnings("ignore:.*")
def test_solid_angle_compare_to_mantid():
    da = _mantid_sans2d_solid_angle_data()
    solid_angle = normalization.solid_angle(
        da,
        **_sans2d_geometry(),
    ).data
    assert sc.allclose(
        da.data['tof', 0], solid_angle, atol=0.0 * sc.Unit('dimensionless')
    )


def test_solid_angle_compare_to_reference_file():
    da = sc.io.load_hdf5(filename=sans2d_solid_angle_reference())
    solid_angle = normalization.solid_angle(
        da,
        sample_position=da.coords['sample_position'],
        **_sans2d_geometry(),
    ).data
    assert sc.allclose(
        da.data['tof', 0], solid_angle, atol=0.0 * sc.Unit('dimensionless')
    )


def test_transmission_fraction():
    N = 100
    wavelength = sc.linspace(
        dim='wavelength', start=2.0, stop=16.0, num=N + 1, unit='angstrom'
    )
    sample_incident_monitor = sc.DataArray(
        data=sc.array(
            dims=['wavelength'], values=100.0 * np.random.random(N), unit='counts'
        ),
        coords={'wavelength': wavelength},
    )
    sample_transmission_monitor = sc.DataArray(
        data=sc.array(
            dims=['wavelength'], values=50.0 * np.random.random(N), unit='counts'
        ),
        coords={'wavelength': wavelength},
    )

    direct_incident_monitor = sc.DataArray(
        data=sc.array(
            dims=['wavelength'], values=100.0 * np.random.random(N), unit='counts'
        ),
        coords={'wavelength': wavelength},
    )
    direct_transmission_monitor = sc.DataArray(
        data=sc.array(
            dims=['wavelength'], values=80.0 * np.random.random(N), unit='counts'
        ),
        coords={'wavelength': wavelength},
    )

    trans_frac = normalization.transmission_fraction(
        sample_incident_monitor=sample_incident_monitor,
        sample_transmission_monitor=sample_transmission_monitor,
        direct_incident_monitor=direct_incident_monitor,
        direct_transmission_monitor=direct_transmission_monitor,
    )

    # If counts on data transmission monitor have increased, it means less neutrons
    # have been absorbed and transmission fraction should increase.
    # - data run: incident: 100 -> transmission: 75
    # - direct run: incident: 100 -> transmission: 80
    assert sc.allclose(
        (trans_frac * sc.scalar(1.5)).data,
        normalization.transmission_fraction(
            sample_incident_monitor=sample_incident_monitor,
            sample_transmission_monitor=sample_transmission_monitor * sc.scalar(1.5),
            direct_incident_monitor=direct_incident_monitor,
            direct_transmission_monitor=direct_transmission_monitor,
        ).data,
    )

    # If counts on direct transmission monitor are higher, it means that many more
    # neutrons are absorbed when the sample is in the path of the beam, and therefore
    # the transmission fraction should decrease.
    # - data run: incident: 100 -> transmission: 50
    # - direct run: incident: 100 -> transmission: 90
    assert sc.allclose(
        (trans_frac / sc.scalar(9 / 8)).data,
        normalization.transmission_fraction(
            sample_incident_monitor=sample_incident_monitor,
            sample_transmission_monitor=sample_transmission_monitor,
            direct_incident_monitor=direct_incident_monitor,
            direct_transmission_monitor=direct_transmission_monitor * sc.scalar(9 / 8),
        ).data,
    )

    # If counts on direct incident monitor are higher, but counts on direct transmission
    # monitor are the same, it means that the relative difference between incident and
    # transmission has increased for the direct run, but not for the data run.
    # This would be the case where neutron beam flux was higher during the direct run.
    # This implies that that the transmission fraction is higher than in our vanilla
    # run.
    # - data run: incident: 100 -> transmission: 50
    # - direct run: incident: 110 -> transmission: 80
    assert sc.allclose(
        (trans_frac * sc.scalar(1.1)).data,
        normalization.transmission_fraction(
            sample_incident_monitor=sample_incident_monitor,
            sample_transmission_monitor=sample_transmission_monitor,
            direct_incident_monitor=direct_incident_monitor * sc.scalar(1.1),
            direct_transmission_monitor=direct_transmission_monitor,
        ).data,
    )

    # If counts on data incident monitor are higher, but counts on data transmission
    # monitor are the same, it means that more neutrons were absorbed in this run,
    # and then transmission fraction decreases.
    # - data run: incident: 110 -> transmission: 50
    # - direct run: incident: 100 -> transmission: 80
    assert sc.allclose(
        (trans_frac / sc.scalar(1.1)).data,
        normalization.transmission_fraction(
            sample_incident_monitor=sample_incident_monitor * sc.scalar(1.1),
            sample_transmission_monitor=sample_transmission_monitor,
            direct_incident_monitor=direct_incident_monitor,
            direct_transmission_monitor=direct_transmission_monitor,
        ).data,
    )


@pytest.fixture
def wavelength_bins() -> sc.Variable:
    return sc.linspace('wavelength', 1.0, 13.0, num=51, unit='angstrom')


@pytest.fixture
def q_bins() -> sc.Variable:
    return sc.linspace('Q', 0.0, 1.0, num=2, unit='1/angstrom')


def _flat_density(bins: sc.Variable, q_bins: sc.Variable) -> sc.DataArray:
    """Events with a uniform density of one count per angstrom.

    Reducing this over a band yields the width of the band in angstrom, so any
    discrepancy between two representations is the discrepancy in the wavelength range
    they select. The event count must stay large compared to the number of bands, else
    the discretization of the uniform placement dominates the comparison tolerance.
    """
    n = 120_000
    lo, hi = bins.min().value, bins.max().value
    return sc.DataArray(
        sc.full(dims=['event'], shape=[n], value=(hi - lo) / n, unit='counts'),
        coords={
            'wavelength': sc.array(
                dims=['event'],
                values=np.linspace(lo, hi, n, endpoint=False),
                unit='angstrom',
            ),
            'Q': sc.full(dims=['event'], shape=[n], value=0.5, unit='1/angstrom'),
        },
    ).bin(Q=q_bins, wavelength=bins)


def _as_midpoints(histogram: sc.DataArray) -> sc.DataArray:
    """Replace the wavelength bin edges by midpoints.

    The I(Q) denominator is dense in this form, because computing Q requires one
    wavelength value per bin. See :func:`normalization.norm_detector_term_denominator`.
    """
    return histogram.assign_coords(
        wavelength=sc.midpoints(histogram.coords['wavelength'])
    )


@pytest.mark.parametrize('nbands', [7, 10, 13])
def test_reduce_q_selects_same_wavelength_range_for_all_representations(
    wavelength_bins, q_bins, nbands
):
    bands = normalization.process_wavelength_bands(
        sc.linspace(
            'wavelength',
            wavelength_bins.min().value,
            wavelength_bins.max().value,
            num=nbands + 1,
            unit='angstrom',
        ),
        wavelength_bins,
    )
    events = _flat_density(wavelength_bins, q_bins)
    representations = {
        'events': events,
        'bin_edges': events.hist(),
        'midpoints': _as_midpoints(events.hist()),
    }
    reduced = {
        name: normalization.reduce_q(data, bands=bands)
        for name, data in representations.items()
    }
    reduced['events'] = reduced['events'].hist()
    for name, result in reduced.items():
        assert sc.allclose(result.data, reduced['events'].data, rtol=sc.scalar(1e-4)), (
            name
        )


def test_process_wavelength_bands_returns_exact_bin_edges(wavelength_bins):
    bands = sc.linspace('wavelength', 1.0, 13.0, num=11, unit='angstrom')
    processed = normalization.process_wavelength_bands(bands, wavelength_bins)
    assert sc.identical(
        processed,
        sc.concat([bands[:-1], bands[1:]], dim='x').rename(
            x='wavelength', wavelength='band'
        ),
    )


def test_process_wavelength_bands_snaps_unaligned_bands_onto_bins(wavelength_bins):
    bands = sc.linspace('wavelength', 1.0, 13.0, num=8, unit='angstrom')
    processed = normalization.process_wavelength_bands(bands, wavelength_bins)
    assert set(np.unique(processed.values)) <= set(wavelength_bins.values)
    half_width = 0.5 * (wavelength_bins[1] - wavelength_bins[0]).value
    assert np.all(np.abs(np.unique(processed.values) - bands.values) <= half_width)


def test_process_wavelength_bands_is_idempotent(wavelength_bins):
    """`direct_beam` feeds already-processed bands back in as `WavelengthBands`."""
    bands = sc.linspace('wavelength', 1.0, 13.0, num=8, unit='angstrom')
    once = normalization.process_wavelength_bands(bands, wavelength_bins)
    assert sc.identical(
        normalization.process_wavelength_bands(once, wavelength_bins), once
    )


def test_process_wavelength_bands_snaps_overlapping_bands(wavelength_bins):
    edges = sc.linspace('band', 1.0, 13.0, num=12, unit='angstrom')
    bands = sc.concat([edges[:-2], edges[2:]], dim='wavelength').transpose()
    processed = normalization.process_wavelength_bands(bands, wavelength_bins)
    assert processed.dims == bands.dims
    assert set(np.unique(processed.values)) <= set(wavelength_bins.values)
    bin_width = (wavelength_bins[1] - wavelength_bins[0]).value
    processed_hi = processed['wavelength', 1]['band', :-1]
    processed_lo = processed['wavelength', 0]['band', 1:]
    overlap = (processed_hi - processed_lo).values
    bands_hi = bands['wavelength', 1]['band', :-1]
    bands_lo = bands['wavelength', 0]['band', 1:]
    expected = (bands_hi - bands_lo).values
    assert np.all(np.abs(overlap - expected) <= bin_width)


def test_process_wavelength_bands_raises_if_bands_narrower_than_bins(wavelength_bins):
    bands = sc.linspace('wavelength', 1.0, 13.0, num=201, unit='angstrom')
    with pytest.raises(ValueError, match='collapse to zero width'):
        normalization.process_wavelength_bands(bands, wavelength_bins)


@pytest.mark.parametrize(
    'values', [[0.5, 5.0], [5.0, 20.0], [-3.0, -1.0], [2.0, float('nan')]]
)
def test_process_wavelength_bands_raises_if_bands_outside_bins(wavelength_bins, values):
    bands = sc.array(dims=['wavelength'], values=values, unit='angstrom')
    with pytest.raises(ValueError, match='must lie within'):
        normalization.process_wavelength_bands(bands, wavelength_bins)


def test_process_wavelength_bands_raises_if_band_is_reversed(wavelength_bins):
    bands = sc.array(dims=['band', 'wavelength'], values=[[6.0, 3.0]], unit='angstrom')
    with pytest.raises(ValueError, match='start before they end'):
        normalization.process_wavelength_bands(bands, wavelength_bins)
