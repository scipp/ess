# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Tools for detector calibration."""

from __future__ import annotations

from collections.abc import Callable, ItemsView, Iterable, Iterator, KeysView, Mapping

import scipp as sc
import scipp.constants

from .types import DspacingData, SampleRun


class OutputCalibrationData(Mapping[int, sc.Variable]):
    r"""Calibration data for output ToF data.

    Only one value is stored per coefficient.
    This means that individual detector pixels are *not* resolved but merged
    into average quantities.

    This is a mapping :math:`M` from powers :math:`p` to coefficients :math:`c`
    according to
    .. math::

        t = \sum_{(p, c) \in M}\, c d^p

    where :math:`d` is d-spacing and :math:`t` is time-of-flight.
    """

    def __init__(
        self,
        coefficients: Mapping[int, sc.Variable] | Iterable[tuple[int | sc.Variable]],
    ) -> None:
        self._coefficients = dict(coefficients)

    def __getitem__(self, power: int) -> sc.Variable:
        return self._coefficients[power]

    def __iter__(self) -> Iterator[int]:
        return iter(self._coefficients)

    def __len__(self) -> int:
        return len(self._coefficients)

    def keys(self) -> KeysView[int]:
        return self._coefficients.keys()

    def items(self) -> ItemsView[int, sc.Variable]:
        return self._coefficients.items()

    def __str__(self) -> str:
        return str(self._coefficients)

    def __repr__(self) -> str:
        return f"ScalarCalibrationData({self._coefficients})"

    def d_to_tof_transformer(self) -> Callable[[sc.Variable], sc.Variable]:
        """Return a function to convert d-spacing to ToF."""
        if self.keys() != {1}:
            raise NotImplementedError(
                "Conversion from d-spacing to ToF with calibiration "
                "only supports power 1 (DIFC)."
            )
        difc = self[1]

        def d_to_tof(dspacing: sc.Variable) -> sc.Variable:
            return sc.to_unit(difc * dspacing, unit='us')

        return d_to_tof

    def to_cif_units(self) -> OutputCalibrationData:
        """Convert to the units used in CIF pd_calib_d_to_tof."""

        def unit(p: int) -> sc.Unit:
            base = sc.Unit(f'us / (angstrom^{abs(p)})')
            return sc.reciprocal(base) if p < 0 else base

        return OutputCalibrationData({p: c.to(unit=unit(p)) for p, c in self.items()})

    def to_cif_format(self) -> sc.DataArray:
        """Convert to a data array that can be saved to CIF.

        The return value can be passed to
        :meth:`scippneutron.io.cif.CIF.with_powder_calibration`.
        """
        cal = self.to_cif_units()
        return sc.DataArray(
            sc.array(
                dims=['calibration'], values=[c.value for c in cal.values()], unit=None
            ),
            coords={'power': sc.array(dims=['calibration'], values=list(cal.keys()))},
        )


def assemble_output_calibration(data: DspacingData[SampleRun]) -> OutputCalibrationData:
    """Construct output calibration data from average pixel positions."""
    # Use nanmean because pixels without events have position=NaN.
    average_l = sc.nanmean(data.coords["Ltotal"])
    average_two_theta = sc.nanmean(data.coords["two_theta"])
    difc = sc.to_unit(
        2
        * sc.constants.m_n
        / sc.constants.h
        * average_l
        * sc.sin(0.5 * average_two_theta),
        unit='us / angstrom',
    )
    return OutputCalibrationData({1: difc})


providers = (assemble_output_calibration,)
