# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.testing import assert_allclose

from ess.polarization.correction import correct_for_analyzer


class TransmissionFunction:
    def __call__(
        self, time: sc.Variable, wavelength: sc.Variable, plus_minus: str
    ) -> sc.Variable:
        if plus_minus == 'plus':
            return 10 * time * (2 + wavelength)
        else:
            return 10 * time * (2 - wavelength)

    def apply(self, da: sc.DataArray, plus_minus: str) -> float:
        time = da.coords['time']
        wavelength = da.coords['wavelength']
        return self(time, wavelength, plus_minus)


def test_correct_for_analyzer() -> None:
    time = sc.linspace('event', 1, 10, 10, unit='')
    wavelength = sc.linspace('event', 0.1, 1, 10, unit='')
    events = sc.DataArray(
        sc.arange('event', 10),
        coords={'time': time, 'wavelength': wavelength},
    )
    transmission = TransmissionFunction()

    result = correct_for_analyzer(
        analyzer_up=events[:6], analyzer_down=events[6:], transmission=transmission
    )
    up = result.analyzer_up
    down = result.analyzer_down
    assert up.sizes == {'event': 10}
    assert down.sizes == {'event': 10}
    transmission_plus = transmission(time, wavelength, 'plus')
    transmission_minus = transmission(time, wavelength, 'minus')
    denom = transmission_plus**2 - transmission_minus**2
    assert_allclose(up[:6], events[:6] * transmission_plus[:6] / denom[:6])
    assert_allclose(up[6:], -events[6:] * transmission_minus[6:] / denom[6:])
    assert_allclose(down[:6], -events[:6] * transmission_minus[:6] / denom[:6])
    assert_allclose(down[6:], events[6:] * transmission_plus[6:] / denom[6:])
