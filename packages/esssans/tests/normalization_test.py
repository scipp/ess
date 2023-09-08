# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc

from esssans import normalization


def test_solid_angle():
    l2 = np.arange(1.0, 11.0)
    da = sc.DataArray(
        data=sc.array(dims=['x'], values=l2, unit='counts'),
        coords={'L2': sc.array(dims=['x'], values=l2, unit='m')},
    )
    da.coords['pixel_width'] = sc.scalar(2.0)
    da.coords['pixel_height'] = sc.scalar(3.0)

    solid_angle = normalization.solid_angle_rectangular_approximation(da)

    assert sc.isclose(solid_angle[0].data, solid_angle[1].data * 4).value
    assert sc.isclose(solid_angle[0].data, solid_angle[-1].data * 100).value
    da.coords['pixel_width'] *= 2
    assert sc.allclose(
        solid_angle.data * 2,
        normalization.solid_angle_rectangular_approximation(da).data,
    )
    da.coords['pixel_width'] *= 0.5
    da.coords['pixel_height'] *= 3
    assert sc.allclose(
        solid_angle.data * 3,
        normalization.solid_angle_rectangular_approximation(da).data,
    )


def test_solid_angle_keeps_relevant_masks():
    l2 = np.arange(1.0, 11.0)
    tof = sc.arange('tof', 101, unit='us')
    da = sc.DataArray(
        data=sc.array(
            dims=['spectrum', 'tof'], values=np.random.random([10, 100]), unit='counts'
        ),
        coords={'L2': sc.array(dims=['spectrum'], values=l2, unit='m'), 'tof': tof},
        masks={
            'mask1': sc.array(dims=['spectrum'], values=l2 > 5),
            'mask2': sc.array(dims=['tof'], values=tof < sc.scalar(50, unit='us')),
        },
    )
    da.coords['pixel_width'] = sc.scalar(2.0)
    da.coords['pixel_height'] = sc.scalar(3.0)

    solid_angle = normalization.solid_angle_rectangular_approximation(data=da)

    assert 'mask1' in solid_angle.masks
    assert 'mask2' not in solid_angle.masks


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
