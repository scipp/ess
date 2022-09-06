# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# Author: Simon Heybrock
from os import PathLike
from typing import Dict, Union
import scipp as sc
import scippnexus as snx
from ._classes import Instrument, Entry


def _get_entry(group: snx.NXobject) -> snx.NXentry:
    return group if group.nx_class == snx.NXentry else group.entry


def _load_items(items: Dict[str, snx.NXobject],
                skip_errors: bool = False) -> Dict[str, sc.DataArray]:
    result = {}
    for k, v in items.items():
        try:
            result[k] = v[()]
        except IndexError as e:
            if not skip_errors:
                raise e from None
    return result


def _load(group: snx.NXobject,
          nxclass: type,
          skip_errors: bool = False) -> Dict[str, sc.DataArray]:
    return _load_items(group[nxclass], skip_errors=skip_errors)


def _load_instrument(group: snx.NXobject,
                     skip_errors: bool = False) -> Dict[str, sc.DataArray]:
    instrument = {}
    #if (detectors := _load(group, snx.NXdetector, skip_errors=skip_errors)):
    #    instrument['detectors'] = detectors
    if (disk_choppers := _load(group, snx.NXdisk_chopper, skip_errors=skip_errors)):
        instrument['disk_choppers'] = disk_chopper
    if group[snx.NXsource]:
        instrument['source'] = group.source[()]
    return Instrument(**instrument)


def _load_entry(
        entry: snx.NXentry,
        skip_errors: bool = False) -> Dict[str, Union[Dict, sc.DataArray, sc.Dataset]]:
    content = {}
    if (instrument := _load_instrument(entry.instrument, skip_errors=skip_errors)):
        content['instrument'] = instrument
    #if (monitors := _load(group, snx.NXmonitor, skip_errors=skip_errors)):
    #    content['monitors'] = monitors
    if entry[snx.NXsample]:
        content['sample'] = entry.sample[()]
    return Entry(**content)


def load(filename: Union[str, PathLike]) -> dict:
    with snx.File(filename) as f:
        entry = _load_entry(f.entry, skip_errors=True)
        detectors = _load(f.entry.instrument, snx.NXdetector, skip_errors=True)
        monitors = _load(f.entry, snx.NXmonitor, skip_errors=True)
        entry.instrument.detectors = detectors
        entry.monitors = monitors
    return entry


def load_detectors(filename: Union[str, PathLike]) -> dict:
    with snx.File(filename) as f:
        return _load(f.entry.instrument, snx.NXdetector, skip_errors=True)


def load_monitors(filename: Union[str, PathLike]) -> dict:
    with snx.File(filename) as f:
        return _load(f.entry, snx.NXmonitor, skip_errors=True)


def load_metadata(filename: Union[str, PathLike]) -> dict:
    """Load everything except detectors and monitors, which could be large."""
    with snx.File(filename) as f:
        entry = _load_entry(f.entry)
