# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.testing import assert_allclose

from ess.polarization.correction import compute_polarizing_element_correction


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


def test_compute_polarizing_element_correction() -> None:
    time = sc.linspace('event', 1, 10, 10, unit='')
    wavelength = sc.linspace('event', 0.1, 1, 10, unit='')
    events = sc.DataArray(
        sc.arange('event', 10),
        coords={'time': time, 'wavelength': wavelength},
    )
    transmission = TransmissionFunction()

    result = compute_polarizing_element_correction(
        channel=events, transmission=transmission
    )
    diag = result.diag
    off_diag = result.off_diag
    assert diag.sizes == {'event': 10}
    assert off_diag.sizes == {'event': 10}
    transmission_plus = transmission(time, wavelength, 'plus')
    transmission_minus = transmission(time, wavelength, 'minus')
    denom = transmission_plus**2 - transmission_minus**2
    assert_allclose(diag, transmission_plus / denom)
    assert_allclose(off_diag, -transmission_minus / denom)
