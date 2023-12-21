# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc

from esssans import normalization
from esssans.sans2d.data import get_path

# See https://github.com/mantidproject/mantid/blob/main/instrument/SANS2D_Definition_Tubes.xml  # noqa: E501
_SANS2D_PIXEL_RADIUS = 0.00405 * sc.Unit('m')
_SANS2D_PIXEL_LENGTH = 0.002033984375 * sc.Unit('m')
_SANS2D_SOLID_ANGLE_REFERENCE_FILE = 'SANS2D00063091.SolidAngle_from_mantid.hdf5'


def _sans2d_setup_requirements_for_solid_angle(da):
    '''da.coords contains the pixel positions and the sample position'''
    pixel_positions = da.coords['position'] - da.coords['sample_position']

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

    da = sc.DataArray(
        coords={'position': pixel_positions},
        # Some dummy counts
        data=sc.array(
            dims=['spectrum'],
            values=np.random.randint(0, 100, *pixel_positions.shape),
            unit='counts',
        ),
    )
    return da, pixel_shape, transform


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
        *_sans2d_setup_requirements_for_solid_angle(da)
    ).data
    assert sc.allclose(
        da.data['tof', 0], solid_angle, atol=0.0 * sc.Unit('dimensionless')
    )


def test_solid_angle_compare_to_reference_file():
    da = sc.io.load_hdf5(filename=get_path(_SANS2D_SOLID_ANGLE_REFERENCE_FILE))
    solid_angle = normalization.solid_angle(
        *_sans2d_setup_requirements_for_solid_angle(da)
    ).data
    assert sc.allclose(
        da.data['tof', 0], solid_angle, atol=0.0 * sc.Unit('dimensionless')
    )
