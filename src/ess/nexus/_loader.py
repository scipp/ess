# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# Author: Simon Heybrock
from os import PathLike
from typing import Dict, Union
import scipp as sc
import scippnexus as snx


def _get_entry(group: snx.NXobject) -> snx.NXentry:
    return group if group.nx_class == snx.NXentry else group.entry


def _load_items(items: Dict[str, snx.NXobject]) -> Dict[str, sc.DataArray]:
    return {k: v[()] for k, v in items.items()}


def load_monitors(group: snx.NXobject) -> Dict[str, sc.DataArray]:
    entry = _get_entry(group)
    return _load_items(entry[snx.NXmonitor])
