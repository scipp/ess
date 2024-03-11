# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc

from ess.isissans.data import get_path
from ess.sans import normalization

# See https://github.com/mantidproject/mantid/blob/main/instrument/SANS2D_Definition_Tubes.xml  # noqa: E501
_SANS2D_PIXEL_RADIUS = 0.00405 * sc.Unit('m')
_SANS2D_PIXEL_LENGTH = 0.002033984375 * sc.Unit('m')
_SANS2D_SOLID_ANGLE_REFERENCE_FILE = 'SANS2D00063091.SolidAngle_from_mantid.h5'


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
    transform = sc.spatial.rotation(value=[0, 0, 1 / 2**0.5, 1 / 2**0.5])
    return dict(pixel_shape=pixel_shape, transform=transform)


def _mantid_sans2d_solid_angle_data():
    simpleapi = pytest.importorskip("mantid.simpleapi")
    scippneutron = pytest.importorskip("scippneutron")

    ws = simpleapi.Load(get_path('SANS2D00063091.nxs'))
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
    # sc.io.hdf5.save_hdf5(da, _SANS2D_SOLID_ANGLE_REFERENCE_FILE)
    # Note, also update the name, don't overwrite old reference files.
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
    da = sc.io.load_hdf5(filename=get_path(_SANS2D_SOLID_ANGLE_REFERENCE_FILE))
    solid_angle = normalization.solid_angle(
        da,
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
