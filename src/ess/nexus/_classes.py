# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# Author: Simon Heybrock
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Any
import scipp as sc


# Use dataclasses since dict keys such as `source` or `detectors` might clash with
# other field names

# Nesting is a bad idea for user facing
# Must write loaders in a way that upstream ading new groups/class does not break anything
# -> explicit list of everything we load?
# How can we ensure code<->file compatibility over many years of changes?

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

@dataclass
class Fields:
    fields: dict = field(default_factory=dict)

@dataclass
class Sample:
    sample: Optional[sc.DataArray] = None

@dataclass
class Monitors:
    monitors: dict = field(default_factory=dict)

@dataclass
class Detectors:
    detectors: dict = field(default_factory=dict)

@dataclass
class DiskChoppers:
    disk_choppers: dict = field(default_factory=dict)

@dataclass
class Instrument:
    instrument: Any = None

@dataclass
class GenericInstrument(Detectors, Fields):
    pass

@dataclass
class GenericEntry(Instrument, Sample, Monitors, Fields):
    pass
