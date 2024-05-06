import h5py
import scippnexus as snx


def _load(filepath, *paths: str):
    with snx.File(filepath, 'r') as f:
        for path in paths:
            if path is not None:
                yield f[path][()]
            else:
                yield None


def _convert_nxpaths(filepath, *paths: str):
    with h5py.File(filepath, 'r') as f:
        for path in paths:
            raw_path = []
            current = f
            for step in path.split('/'):
                if step.startswith('NX'):
                    for key, subgroup in current.items():
                        if subgroup.attrs.get('NX_class') == step:
                            current = subgroup
                            raw_path.append(key)
                            break
                else:
                    current = current.get(step)
                    raw_path.append(step)
                    if current is None:
                        yield None
                        break
            else:
                yield '/'.join(raw_path)


def load_nx(filepath, *paths: str):
    '''Helper that loads data from nexus file
    Takes a number of paths of the form 'NXentry/NXuser/name'
    where each part of the path can be either a NXclass or a key.
    Any parts starting with 'NX' are assumed to be nexus classes
    and all other parts are assumed to be keys.
    '''
    yield from _load(filepath, *_convert_nxpaths(filepath, *paths))
