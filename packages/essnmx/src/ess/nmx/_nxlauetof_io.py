# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import warnings

import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import FilePath, NeXusFile

from .types import ControlMode


def _validate_entry(entry: snx.Group) -> None:
    if str(entry.attrs['NX_class']) != 'NXlauetof':
        raise ValueError("File entry is not NXlauetof.")
    _MANDATORY_FIELDS = ('control', 'instrument', 'sample')
    missing_fields = [field for field in _MANDATORY_FIELDS if field not in entry]
    if any(missing_fields):
        raise ValueError("File entry missing mandatory fields, ", missing_fields)


def _as_vector(var: sc.Variable) -> sc.Variable:
    if var.dims == () and var.dtype == sc.DType.vector3:
        return var
    elif len(var.dims) == 1 and var.sizes[var.dim] == 3:
        return sc.vector(value=var.values, unit=var.unit)
    else:
        warnings.warn(
            f"Cannot convert to vector3 scalar: {var}. "
            "Falling back to the original form.",
            UserWarning,
            stacklevel=3,
        )
        return var


def _handle_sample(sample: snx.Group) -> sc.DataGroup:
    sample_dg = sample[...]
    sample_dg['crystal_rotation'] = _as_vector(sample_dg['crystal_rotation'])
    sample_dg['position'] = _as_vector(sample_dg['position'])
    unit_cell = sample_dg.pop('unit_cell')
    sample_dg['unit_cell_length'] = sc.vector(
        unit_cell[:3], unit=sample['unit_cell'].attrs['length-unit']
    )
    sample_dg['unit_cell_angle'] = sc.vector(
        unit_cell[3:], unit=sample['unit_cell'].attrs['angle-unit']
    )
    return sample_dg


def _handle_monitor(control_dg: sc.DataGroup, control: snx.Group) -> sc.DataGroup:
    tof_bin_coord_key = 'tof_bin_coord'

    if tof_bin_coord_key in control.attrs:
        tof_bin_coord = control.attrs['tof_bin_coord']
        control_dg['tof_bin_coord'] = tof_bin_coord
        data: sc.DataArray = control_dg['data']
        data.coords[tof_bin_coord] = data.coords.pop('time_of_flight')

    control_dg['mode'] = ControlMode[control_dg['mode']]

    return control_dg


def load_essnmx_nxlauetof(file: str | FilePath | NeXusFile) -> sc.DataGroup:
    dg = snx.load(file)

    with snx.File(file) as f:
        _validate_entry(entry := f['entry'])
        dg['entry']['sample'] = _handle_sample(entry['sample'])
        dg['entry']['control'] = _handle_monitor(
            dg['entry']['control'], entry['control']
        )

    return dg['entry']
