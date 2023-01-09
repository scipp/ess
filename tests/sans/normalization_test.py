# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc
from ess.sans import normalization


def test_solid_angle():
    l2 = np.arange(1., 11.)
    da = sc.DataArray(data=sc.array(dims=['x'], values=l2, unit='counts'),
                      coords={'L2': sc.array(dims=['x'], values=l2, unit='m')})
    pixel_width = 2.0
    pixel_height = 3.0

    solid_angle = normalization.solid_angle_of_rectangular_pixels(
        data=da, pixel_width=pixel_width, pixel_height=pixel_height)

    assert sc.isclose(solid_angle[0], solid_angle[1] * 4).value
    assert sc.isclose(solid_angle[0], solid_angle[-1] * 100).value
    assert sc.allclose(
        solid_angle * 2,
        normalization.solid_angle_of_rectangular_pixels(data=da,
                                                        pixel_width=pixel_width * 2,
                                                        pixel_height=pixel_height))
    assert sc.allclose(
        solid_angle * 3,
        normalization.solid_angle_of_rectangular_pixels(data=da,
                                                        pixel_width=pixel_width,
                                                        pixel_height=pixel_height * 3))


def test_transmission_fraction():
    N = 100
    wavelength = sc.linspace(dim='wavelength',
                             start=2.0,
                             stop=16.0,
                             num=N + 1,
                             unit='angstrom')
    data_monitors = {
        'incident':
        sc.DataArray(data=sc.array(dims=['wavelength'],
                                   values=100.0 * np.random.random(N),
                                   unit='counts'),
                     coords={'wavelength': wavelength}),
        'transmission':
        sc.DataArray(data=sc.array(dims=['wavelength'],
                                   values=50.0 * np.random.random(N),
                                   unit='counts'),
                     coords={'wavelength': wavelength})
    }

    direct_monitors = {
        'incident':
        sc.DataArray(data=sc.array(dims=['wavelength'],
                                   values=100.0 * np.random.random(N),
                                   unit='counts'),
                     coords={'wavelength': wavelength}),
        'transmission':
        sc.DataArray(data=sc.array(dims=['wavelength'],
                                   values=80.0 * np.random.random(N),
                                   unit='counts'),
                     coords={'wavelength': wavelength})
    }

    trans_frac = normalization.transmission_fraction(data_monitors=data_monitors,
                                                     direct_monitors=direct_monitors)

    # If counts on data transmission monitor have increased, it means less neutrons
    # have been absorbed and transmission fraction should increase.
    # - data run: incident: 100 -> transmission: 75
    # - direct run: incident: 100 -> transmission: 80
    assert sc.allclose(
        (trans_frac * sc.scalar(1.5)).data,
        normalization.transmission_fraction(data_monitors={
            'incident':
            data_monitors['incident'],
            'transmission':
            data_monitors['transmission'] * sc.scalar(1.5)
        },
                                            direct_monitors=direct_monitors).data)

    # If counts on direct transmission monitor are higher, it means that many more
    # neutrons are absorbed when the sample is in the path of the beam, and therefore
    # the transmission fraction should decrease.
    # - data run: incident: 100 -> transmission: 50
    # - direct run: incident: 100 -> transmission: 90
    assert sc.allclose((trans_frac / sc.scalar(9 / 8)).data,
                       normalization.transmission_fraction(
                           data_monitors=data_monitors,
                           direct_monitors={
                               'incident':
                               direct_monitors['incident'],
                               'transmission':
                               direct_monitors['transmission'] * sc.scalar(9 / 8)
                           }).data)

    # If counts on direct incident monitor are higher, but counts on direct transmission
    # monitor are the same, it means that the relative difference between incident and
    # transmission has increased for the direct run, but not for the data run.
    # This would be the case where neutron beam flux was higher during the direct run.
    # This implies that that the transmission fraction is higher than in our vanilla
    # run.
    # - data run: incident: 100 -> transmission: 50
    # - direct run: incident: 110 -> transmission: 80
    assert sc.allclose((trans_frac * sc.scalar(1.1)).data,
                       normalization.transmission_fraction(
                           data_monitors=data_monitors,
                           direct_monitors={
                               'incident': direct_monitors['incident'] * sc.scalar(1.1),
                               'transmission': direct_monitors['transmission']
                           }).data)

    # If counts on data incident monitor are higher, but counts on data transmission
    # monitor are the same, it means that more neutrons were absorbed in this run,
    # and then transmission fraction decreases.
    # - data run: incident: 110 -> transmission: 50
    # - direct run: incident: 100 -> transmission: 80
    assert sc.allclose(
        (trans_frac / sc.scalar(1.1)).data,
        normalization.transmission_fraction(data_monitors={
            'incident':
            data_monitors['incident'] * sc.scalar(1.1),
            'transmission':
            data_monitors['transmission']
        },
                                            direct_monitors=direct_monitors).data)
