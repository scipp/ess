import numpy as np
import scipp as sc


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
            data.append(list(map(float, line.strip().split(' '))))

    data = np.array(data)

    if 'ylabel' in meta:
        labels = meta['ylabel'].strip().split(' ')
        if labels[0] == 'p':
            da = sc.DataArray(
                sc.array(dims=['events'], values=data[:, 0], variances=data[:, 0] ** 2),
                coords={
                    label: sc.array(dims=['events'], values=values)
                    for values, label in zip(data[:, 1:].T, labels[1:], strict=False)
                },
            )
            for k, v in meta.items():
                da.coords[k] = sc.scalar(v)
            return da
    raise ValueError('Could not parse the file as a list of events.')


def parse_events_h5(f):
    pass


'''
def parse_events_h5(f):
    f['entry1/data']
    events = load_nx(f, 'NXentry/NXdetector/NXdata')
    parameters = load_nx(f, 'NXentry/simulation/Param')
'''
