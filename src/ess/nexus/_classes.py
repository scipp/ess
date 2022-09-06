# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# Author: Simon Heybrock
from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import scipp as sc


@dataclass
class Instrument:
    source: Dict[str, Union[sc.Variable, sc.DataArray]] = field(default_factory=dict)
    detectors: Dict[str, sc.DataArray] = field(default_factory=dict)
    disk_choppers: Dict[str, Dict[str,
                                  Union[sc.Variable,
                                        sc.DataArray]]] = field(default_factory=dict)


@dataclass
class Entry:
    instrument: Optional[Instrument] = None
    monitors: Dict[str, sc.DataArray] = field(default_factory=dict)
    sample: Dict[str, Union[sc.Variable, sc.DataArray]] = field(default_factory=dict)
