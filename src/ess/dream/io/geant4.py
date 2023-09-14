# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import os
from typing import Dict, Optional, Union

import scipp as sc

MANTLE_DETECTOR_ID = sc.index(7)
HIGH_RES_DETECTOR_ID = sc.index(8)
ENDCAPS_DETECTOR_IDS = tuple(map(sc.index, (3, 4, 5, 6)))


def load_geant4_csv(filename: Union[str, os.PathLike]) -> sc.DataGroup:
    """Load a GEANT4 CSV file for DREAM.

    Parameters
    ----------
    filename:
        Path to the GEANT4 CSV file.

    Returns
    -------
    :
        A :class:`scipp.DataGroup` containing the loaded events.
    """
    events = _load_raw_events(filename)
    detectors = _split_detectors(events)
    for det in detectors.values():
        _adjust_coords(det)

    dg = sc.DataGroup({'instrument': sc.DataGroup(detectors)})
    return _group(dg)


def _load_raw_events(filename: Union[str, os.PathLike]) -> sc.DataArray:
    table = sc.io.load_csv(filename, sep='\t', header_parser='bracket', data_columns=[])
    table = table.rename_dims(row='event')
    return sc.DataArray(sc.ones(sizes=table.sizes), coords=table.coords)


def _adjust_coords(da: sc.DataArray) -> None:
    da.coords['wavelength'] = da.coords.pop('lambda')
    da.coords['position'] = sc.spatial.as_vectors(
        da.coords.pop('x_pos'), da.coords.pop('y_pos'), da.coords.pop('z_pos')
    )


def _group(dg: sc.DataGroup) -> sc.DataGroup:
    return dg.group('counter', 'segment', 'module', 'strip', 'wire')


def _split_detectors(
    data: sc.DataArray, detector_id_name: str = 'det ID'
) -> Dict[str, sc.DataArray]:
    groups = data.group(
        sc.concat(
            [MANTLE_DETECTOR_ID, HIGH_RES_DETECTOR_ID, *ENDCAPS_DETECTOR_IDS],
            dim=detector_id_name,
        )
    )
    mantle = _extract_detector(groups, detector_id_name, MANTLE_DETECTOR_ID).copy()
    high_res = _extract_detector(groups, detector_id_name, HIGH_RES_DETECTOR_ID).copy()

    endcaps_list = [
        det
        for i in ENDCAPS_DETECTOR_IDS
        if (det := _extract_detector(groups, detector_id_name, i)) is not None
    ]
    if endcaps_list:
        endcaps = sc.concat(endcaps_list, data.dim)
        endcap_forward = endcaps[endcaps.coords['z_pos'] > sc.scalar(0, unit='mm')]
        endcap_backward = endcaps[endcaps.coords['z_pos'] < sc.scalar(0, unit='mm')]
    else:
        endcap_forward = None
        endcap_backward = None

    return {
        key: val
        for key, val in zip(
            ('mantle', 'high_resolution', 'endcap_forward', 'endcap_backward'),
            (mantle, high_res, endcap_forward, endcap_backward),
        )
        if val is not None
    }


def _extract_detector(
    detector_groups: sc.DataArray, detector_id_name: str, detector_id: sc.Variable
) -> Optional[sc.DataArray]:
    try:
        return detector_groups[detector_id_name, detector_id].value
    except IndexError:
        return None
