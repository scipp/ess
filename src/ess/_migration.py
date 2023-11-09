# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import MutableMapping

import scipp as sc


def get_attrs(da: sc.DataArray) -> MutableMapping[str, sc.Variable]:
    try:
        # During deprecation phase
        return da.deprecated_attrs
    except AttributeError:
        try:
            # Before deprecation phase
            return da.attrs
        except AttributeError:
            # After deprecation phase / removal of attrs
            return da.coords


def get_meta(da: sc.DataArray) -> MutableMapping[str, sc.Variable]:
    try:
        # During deprecation phase
        return da.deprecated_meta
    except AttributeError:
        try:
            # Before deprecation phase
            return da.meta
        except AttributeError:
            # After deprecation phase / removal of attrs
            return da.coords
