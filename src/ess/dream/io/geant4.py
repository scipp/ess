# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import os
from io import BytesIO, StringIO
from typing import Dict, Optional, Union

import numpy as np
import scipp as sc

MANTLE_DETECTOR_ID = sc.index(7)
HIGH_RES_DETECTOR_ID = sc.index(8)
ENDCAPS_DETECTOR_IDS = tuple(map(sc.index, (3, 4, 5, 6)))


def load_geant4_csv(
    filename: Union[str, os.PathLike, StringIO, BytesIO]
) -> sc.DataGroup:
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
    detectors = _group(detectors)

    return sc.DataGroup({'instrument': sc.DataGroup(detectors)})


def _load_raw_events(
    filename: Union[str, os.PathLike, StringIO, BytesIO]
) -> sc.DataArray:
    table = sc.io.load_csv(filename, sep='\t', header_parser='bracket', data_columns=[])
    table = table.rename_dims(row='event')
    return sc.DataArray(
        sc.ones(sizes=table.sizes, with_variances=True, unit='counts'),
        coords=table.coords,
    )


def _adjust_coords(da: sc.DataArray) -> None:
    da.coords['wavelength'] = da.coords.pop('lambda')
    da.coords['position'] = sc.spatial.as_vectors(
        da.coords.pop('x_pos'), da.coords.pop('y_pos'), da.coords.pop('z_pos')
    )


def _group(detectors: Dict[str, sc.DataArray]) -> Dict[str, sc.DataArray]:
    elements = ('module', 'segment', 'counter', 'wire', 'strip')

    def group(key: str, da: sc.DataArray) -> sc.DataArray:
        if key == 'high_resolution':
            # Only the HR detector has sectors.
            return da.group('sector', *elements)
        res = da.group(*elements)
        res.bins.coords.pop('sector', None)
        return res

    return {key: group(key, da) for key, da in detectors.items()}


def _split_detectors(
    data: sc.DataArray, detector_id_name: str = 'det ID'
) -> Dict[str, sc.DataArray]:
    groups = data.group(
        sc.concat(
            [MANTLE_DETECTOR_ID, HIGH_RES_DETECTOR_ID, *ENDCAPS_DETECTOR_IDS],
            dim=detector_id_name,
        )
    )
    detectors = {}
    if (
        mantle := _extract_detector(groups, detector_id_name, MANTLE_DETECTOR_ID)
    ) is not None:
        detectors['mantle'] = mantle.copy()
    if (
        high_res := _extract_detector(groups, detector_id_name, HIGH_RES_DETECTOR_ID)
    ) is not None:
        detectors['high_resolution'] = high_res.copy()

    endcaps_list = [
        det
        for i in ENDCAPS_DETECTOR_IDS
        if (det := _extract_detector(groups, detector_id_name, i)) is not None
    ]
    if endcaps_list:
        endcaps = sc.concat(endcaps_list, data.dim)
        endcaps = endcaps.bin(
            z_pos=sc.array(
                dims=['z_pos'],
                values=[-np.inf, 0.0, np.inf],
                unit=endcaps.coords['z_pos'].unit,
            )
        )
        detectors['endcap_backward'] = endcaps[0].bins.concat().value.copy()
        detectors['endcap_forward'] = endcaps[1].bins.concat().value.copy()

    return detectors


def _extract_detector(
    detector_groups: sc.DataArray, detector_id_name: str, detector_id: sc.Variable
) -> Optional[sc.DataArray]:
    try:
        return detector_groups[detector_id_name, detector_id].value
    except IndexError:
        return None
