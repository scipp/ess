from scipp import DataArray, Variable


def norm(vector: Variable) -> Variable:
    from scipp import DType, dot, sqrt

    assert (
        vector.dtype == DType.vector3
    )  # "Vector operations require scipp.DType.vector3 elements!"
    return sqrt(dot(vector, vector))


def in_same_unit(b: Variable, to: Variable | None = None) -> Variable:
    def unit(x):
        if x.bins is not None:
            return x.bins.unit
        return x.unit

    if to is None:
        raise ValueError("The to unit-full object must be specified")

    a_unit = unit(to)
    b_unit = unit(b)
    if a_unit is None and b_unit is None:
        return b
    if a_unit is None or b_unit is None:
        raise ValueError(f"Can not find the units to use for {b} from {to}")
    if a_unit != b_unit:
        b = b.to(unit=a_unit)
    return b


def is_in_coords(x: DataArray, name: str):
    return name in x.coords or (x.bins is not None and name in x.bins.coords)


def split_setting(*args, arg_coord: str | list[str] | None = None, **kwargs):
    """For each of the DataArrays in args, find their segments according to the provided kwargs lookup tables

    Parameters
    ----------
    args:
        The DataArrays for which to perform coordinate lookup
    arg_coord:
        The coordinate name(s) in each DataArray which are used in the lookup
    kwargs:
        The named scipp lookup tables used in the conversion

    The coordinate name to use in the lookup might differ depending on the type/source of DataArray
    so it should be provided in the keyword argument `lookup_coord` as a list, one name per data array

    >>> split_setting(triplets, monitor['first'], arg_coord=['event_time_zero', 'time'], a3=table(a3_log['value'], 'time'))
    """
    from collections import namedtuple
    from itertools import product

    if arg_coord is None:
        raise ValueError("The argument lookup coordinate name(s) must be provided")
    if isinstance(arg_coord, str):
        arg_coord = [arg_coord]
    if len(arg_coord) == 1 and len(args) > 1:
        arg_coord = arg_coord * len(args)

    for arg, coord in zip(args, arg_coord):
        if not is_in_coords(arg, coord):
            raise ValueError(f"No {coord} coordinate in DataArray\n{arg}")

    def dim_size(k, v):
        size = 0
        for argument, coordinate in zip(args, arg_coord):
            a = argument.transform_coords(
                k, graph={k: lambda x: v[x], 'x': coordinate}, keep_aliases=False
            )
            if a.bins is not None:
                a = a.group(k)
            if size and a.sizes[k] and size != a.sizes[k]:
                raise ValueError(
                    f"Earlier-found {k} size, {size}, not consistent with current {a.sizes[k]}"
                )
            elif size == 0:
                size = a.sizes[k]
        return size

    dims = {k: dim_size(k, v) for k, v in kwargs.items()}
    key = namedtuple('key', dims.keys())
    keys = list(dims.keys())
    ranges = [range(x) for x in dims.values()]
    entries = [
        key(**{k: v for k, v in zip(keys, values)}) for values in product(*ranges)
    ]

    graph = {k: lambda x: v[x] for k, v in kwargs.items()}
    targets = tuple(graph.keys())
    # everything = lambda x: tuple(v[x] for v in kwargs.values())

    new_args = [a for a in args]
    for i, coord in enumerate(arg_coord):
        graph['x'] = coord
        new_args[i] = new_args[i].transform_coords(
            targets, graph=graph, keep_aliases=False
        )
        if new_args[i].bins is not None:
            new_args[i] = new_args[i].group(*targets)

    return new_args

    def one_entry(new_arg, entry):
        a = new_arg
        for index, k in enumerate(kwargs.keys()):
            a = a[k, entry[index]]
        return a

    data = {entry: tuple(one_entry(x, entry) for x in new_args) for entry in entries}
    return data
