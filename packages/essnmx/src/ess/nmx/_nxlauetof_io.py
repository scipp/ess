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


def _handle_sample(sample_dg: sc.DataGroup, sample: snx.Group) -> sc.DataGroup:
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


def _handle_source(instrument_dg: sc.DataGroup, instrument: snx.Group) -> sc.DataGroup:
    distance = instrument_dg['source'].pop('distance')
    position = sc.vector(
        instrument['source']['distance'].attrs['position'], unit=distance.unit
    )
    instrument_dg['source']['position'] = position


def _restore_positions(
    *, metadatas: sc.DataGroup, fast_axis_dim: str, slow_axis_dim: str, sizes: dict
) -> sc.Variable:
    fast_axis = metadatas['fast_axis']
    fast_axis_size = sizes[fast_axis_dim]
    slow_axis = metadatas['slow_axis']
    slow_axis_size = sizes[slow_axis_dim]

    pixel_sizes = {
        'x_pixel_offset': metadatas['x_pixel_size'],
        'y_pixel_offset': metadatas['y_pixel_size'],
    }

    fast_axis_offsets = (
        sc.arange(dim=fast_axis_dim, start=0.0, stop=fast_axis_size)
        * pixel_sizes[fast_axis_dim]
        * fast_axis
    )
    slow_axis_offsets = (
        sc.arange(dim=slow_axis_dim, start=0.0, stop=slow_axis_size)
        * pixel_sizes[slow_axis_dim]
        * slow_axis
    )
    # The slow axis should be the outer most dimension.
    detector_sizes = {slow_axis_dim: slow_axis_size, fast_axis_dim: fast_axis_size}

    pixel_offsets = fast_axis_offsets.broadcast(
        sizes=detector_sizes
    ) + slow_axis_offsets.broadcast(sizes=detector_sizes)

    detetor_center = metadatas['origin']
    slow_axis_width = pixel_sizes[slow_axis_dim] * slow_axis_size
    fast_axis_width = pixel_sizes[fast_axis_dim] * fast_axis_size
    detector_corner = (
        detetor_center
        - (slow_axis_width / 2) * slow_axis
        + pixel_sizes[slow_axis_dim] * slow_axis / 2
        - (fast_axis_width / 2) * fast_axis
        + pixel_sizes[fast_axis_dim] * fast_axis / 2
    )

    return pixel_offsets + detector_corner


def _handle_detector_data(
    instrument_dg: sc.DataGroup, instrument: snx.Group
) -> sc.DataGroup:
    detectors: sc.DataGroup[sc.DataGroup] = sc.DataGroup(
        {
            det_name: instrument_dg.pop(det_name)
            for det_name in instrument[snx.NXdetector].keys()
        }
    )
    instrument_dg['detectors'] = detectors
    time_coord_name = next(iter({'tof', 'event_time_offset'} & set(detectors.dims)))

    for det_name, det_gr in detectors.items():
        all_keys = list(filter(lambda key: key != 'data', det_gr.keys()))
        metadatas = sc.DataGroup()
        for key in all_keys:
            metadatas[key] = det_gr.pop(key)

        for vector_field in ('slow_axis', 'fast_axis', 'origin'):
            metadatas[vector_field] = _as_vector(metadatas[vector_field])

        det_gr['metadata'] = metadatas
        slow_axis_dim = instrument[det_name]['slow_axis'].attrs['dim']
        fast_axis_dim = instrument[det_name]['fast_axis'].attrs['dim']
        det_gr['data'] = sc.DataArray(
            data=det_gr['data'],
            coords={
                time_coord_name: metadatas.pop('original_time_edges'),
                'position': _restore_positions(
                    metadatas=metadatas,
                    fast_axis_dim=fast_axis_dim,
                    slow_axis_dim=slow_axis_dim,
                    sizes=det_gr['data'].sizes,
                ),
            },
        )


def load_essnmx_nxlauetof(file: str | FilePath | NeXusFile) -> sc.DataGroup:
    dg = snx.load(file)

    with snx.File(file, mode='r') as f:
        _validate_entry(entry := f['entry'])
        _handle_sample(dg['entry']['sample'], entry['sample'])
        _handle_monitor(dg['entry']['control'], entry['control'])
        _handle_source(dg['entry']['instrument'], entry['instrument'])
        _handle_detector_data(dg['entry']['instrument'], entry['instrument'])

    return dg['entry']
