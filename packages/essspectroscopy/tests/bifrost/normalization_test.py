# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
import scipp.testing
from ess.bifrost.normalization import normalize_by_monitor_and_proton_charge
from ess.spectroscopy.types import UncertaintyBroadcastMode


@pytest.fixture
def detector() -> sc.DataArray:
    events = sc.DataArray(
        sc.ones(dims=['event'], shape=[3], unit='counts', with_variances=True),
        coords={
            'incident_wavelength': sc.array(
                dims=['event'], values=[1.5, 2.5, 3.5], unit='angstrom'
            )
        },
    )
    return sc.DataArray(
        sc.bins(data=events, dim='event', begin=sc.index(0), end=sc.index(3))
    )


@pytest.fixture(params=[False, True], ids=['scalar', 'time-dependent'])
def event_monitor(request: pytest.FixtureRequest) -> sc.DataArray:
    events = sc.DataArray(
        sc.ones(dims=['event'], shape=[4], unit='counts', with_variances=True),
        coords={
            'incident_wavelength': sc.array(
                dims=['event'], values=[1.5, 1.5, 2.5, 3.5], unit='angstrom'
            )
        },
    )
    if request.param:
        events.coords['time'] = sc.array(dims=['event'], values=[0, 1, 0, 1], unit='s')
        return events.group('time')
    return sc.DataArray(
        sc.bins(data=events, dim='event', begin=sc.index(0), end=sc.index(4))
    )


@pytest.mark.parametrize(
    ('edges', 'expected_weights'),
    [
        ([1.0, 3.0, 4.0], [4 / 3, 4 / 3, 2.0]),
        ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 2.0]),
    ],
)
def test_normalize_monitor(
    detector: sc.DataArray,
    event_monitor: sc.DataArray,
    edges: list[float],
    expected_weights: list[float],
) -> None:
    monitor = event_monitor.hist(
        incident_wavelength=sc.array(
            dims=['incident_wavelength'], values=edges, unit='angstrom'
        )
    )

    result = normalize_by_monitor_and_proton_charge(
        detector=detector,
        monitor=monitor,
        proton_charge=sc.DataArray(sc.scalar(2.0, unit='pC')),
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.drop,
    )

    sc.testing.assert_allclose(
        result.bins.constituents['data'].data,
        sc.array(
            dims=['event'],
            values=expected_weights,
            variances=np.square(expected_weights),
            unit='counts*angstrom/pC',
        ),
    )


def test_event_monitor_requires_bins(
    detector: sc.DataArray, event_monitor: sc.DataArray
) -> None:
    with pytest.raises(
        ValueError, match=r'MonitorWavelengthBins\[NormalizationMonitor\]'
    ):
        normalize_by_monitor_and_proton_charge(
            detector=detector,
            monitor=event_monitor,
            proton_charge=sc.DataArray(sc.scalar(2.0, unit='pC')),
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.drop,
        )
