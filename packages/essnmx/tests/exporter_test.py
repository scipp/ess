# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import io

import numpy as np
import pytest
import scipp as sc
from ess.nmx.nexus import export_as_nexus
from ess.nmx.reduction import NMXReducedData


@pytest.fixture()
def reduced_data() -> NMXReducedData:
    rng = np.random.default_rng(42)
    id_list = sc.array(dims=['event'], values=rng.integers(0, 12, size=100))
    t_list = sc.array(dims=['event'], values=rng.random(size=100, dtype=float))
    counts = (
        sc.DataArray(
            data=sc.ones(dims=['event'], shape=[100]),
            coords={'id': id_list, 't': t_list},
        )
        .group('id')
        .hist(t=10)
    )

    return NMXReducedData(
        sc.DataGroup(
            dict(  # noqa: C408
                counts=counts,
                proton_charge=sc.scalar(1.0, unit='counts'),
                crystal_rotation=sc.vector(value=[0.0, 20.0, 0.0], unit='deg'),
                fast_axis=sc.vectors(
                    dims=['panel'],
                    values=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    unit='m',
                ),
                slow_axis=sc.vectors(
                    dims=['panel'],
                    values=[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    unit='m',
                ),
                origin_position=sc.vectors(
                    dims=['panel'],
                    values=[[-0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                    unit='m',
                ),
                sample_position=sc.vector(value=[0.0, 0.0, 0.0], unit='m'),
                source_position=sc.vector(value=[-3, 0.0, -4], unit='m'),
                sample_name=sc.scalar('Unit Test Sample'),
                position=sc.zeros(dims=['panel', 'id'], shape=[3, 4], unit='m'),
            )
        )
    )


def test_mcstas_reduction_export_to_bytestream(reduced_data: NMXReducedData) -> None:
    """Test export method."""
    import h5py
    import numpy as np
    import scipp as sc

    data_fields = [
        'NXdetector',
        'NXsample',
        'NXsource',
        'NXinstrument',
        'definition',
        'name',
    ]

    with io.BytesIO() as bio:
        export_as_nexus(reduced_data, bio)
        with h5py.File(bio, 'r') as f:
            assert 'NMX_data' in f
            nmx_data: h5py.Group = f.require_group('NMX_data')
            for field in data_fields:
                assert field in nmx_data

            nx_detector = nmx_data.require_group('NXdetector')
            assert np.all(
                nx_detector['fast_axis'][()] == reduced_data['fast_axis'].values
            )
            assert np.all(
                nx_detector['slow_axis'][()] == reduced_data['slow_axis'].values
            )
            assert np.all(
                nx_detector['origin'][()] == reduced_data['origin_position'].values
            )

            instrument_data = nmx_data.require_group('NXinstrument')
            assert (
                instrument_data['proton_charge'][()]
                == reduced_data['proton_charge'].value
            )

            det1_data = instrument_data.require_group('detector_1')
            assert np.all(det1_data['counts'][()] == reduced_data['counts'].values)
            assert np.all(
                det1_data['pixel_id'][()] == reduced_data['counts'].coords['id'].values
            )
            assert np.all(
                det1_data['t_bin'][()] == reduced_data['counts'].coords['t'].values
            )

            nx_sample = nmx_data.require_group('NXsample')
            sample_name: bytes = nx_sample['name'][()]
            assert sample_name.decode() == reduced_data['sample_name'].value

            nx_source = nmx_data.require_group('NXsource')
            assert (
                nx_source['distance'][()]
                == sc.norm(reduced_data['source_position']).value
            )
