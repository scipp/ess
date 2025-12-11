# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import h5py
import numpy as np
import pandas as pd
import scipp as sc

from ess.reflectometry.load import load_h5


def parse_metadata_ascii(lines):
    data = {}
    section = None
    for line in lines:
        if line.startswith('begin'):
            _, _, name = line.partition(' ')
            section = {}
        elif line.startswith('end'):
            data.setdefault(name.strip(), []).append(section)
            section = None
        else:
            if section is not None:
                key, _, value = line.partition(': ')
                section[key.strip()] = value.strip()
    return data


def parse_events_ascii(lines):
    meta = {}
    data = []
    for line in lines:
        if line.startswith('#'):
            key, _, value = line[2:].partition(': ')
            if '=' in value:
                key, _, value = value.partition('=')
            meta[key] = value
        else:
            break

    data = pd.read_csv(lines, comment='#', header=None, delimiter=' ')

    if 'ylabel' in meta:
        labels = meta['ylabel'].strip().split(' ')
        if labels[0] == 'p':
            da = sc.DataArray(
                # The squares on the variances is the correct way
                # to load weighted event data.
                # Consult the McStas documentation
                # (section 2.2.1) https://www.mcstas.org/documentation/manual/
                # for more information.
                sc.array(
                    dims=['events'],
                    values=data[0].to_numpy(),
                    variances=data[0].to_numpy() ** 2,
                ),
                coords={
                    label: sc.array(dims=['events'], values=data[i].to_numpy())
                    for i, label in enumerate(labels)
                    if i != 0
                },
            )
            for k, v in meta.items():
                da.coords[k] = sc.scalar(v)
            return da
    raise ValueError('Could not parse the file as a list of events.')


def parse_events_h5(f, events_to_sample_per_unit_weight=None):
    if isinstance(f, str):
        with h5py.File(f) as ff:
            return parse_events_h5(ff)

    data, events, params = load_h5(
        f,
        'NXentry/NXdetector/NXdata',
        'NXentry/NXdetector/NXdata/events',
        'NXentry/simulation/Param',
    )
    events = events[()]
    if events_to_sample_per_unit_weight is None:
        da = sc.DataArray(
            # The squares on the variances is the correct way to load
            # weighted event data.
            # Consult the McStas documentation
            # (section 2.2.1) https://www.mcstas.org/documentation/manual/
            # for more information.
            sc.array(dims=['events'], values=events[:, 0], variances=events[:, 0] ** 2),
        )
        for i, label in enumerate(data.attrs["ylabel"].decode().strip().split(' ')):
            if label == 'p':
                continue
            da.coords[label] = sc.array(dims=['events'], values=events[:, i])
    else:
        weights = events[:, 0]
        total_weight = weights.sum()
        nevents_to_sample = round(events_to_sample_per_unit_weight * total_weight)
        inds = np.random.choice(
            np.arange(len(weights)),
            nevents_to_sample,
            p=weights / total_weight,
        )
        da = sc.DataArray(
            sc.ones(dims=['events'], shape=(nevents_to_sample,), with_variances=True),
        )
        for i, label in enumerate(data.attrs["ylabel"].decode().strip().split(' ')):
            if label == 'p':
                continue
            da.coords[label] = sc.array(dims=['events'], values=events[inds, i])

    for name, value in data.attrs.items():
        da.coords[name] = sc.scalar(value.decode())

    for k, v in params.items():
        v = v[0]
        if isinstance(v, bytes):
            v = v.decode()
        da.coords[k] = sc.scalar(v)
    return da
