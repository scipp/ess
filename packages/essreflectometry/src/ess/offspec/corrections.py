# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc


def correct_by_monitor(
    da: sc.DataArray,
    mon: sc.DataArray,
    wlims: tuple[sc.Variable, sc.Variable],
    wbmin: sc.Variable,
) -> sc.DataArray:
    "Corrects the data by the monitor intensity"
    mon = mon - sc.values(mon['wavelength', wbmin:].mean())
    return da / sc.values(mon['wavelength', wlims[0] : wlims[-1]].sum())
