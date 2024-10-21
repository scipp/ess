# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from scipp import DataArray, Variable


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


def range_normalized(variable: Variable, minimum: Variable, maximum: Variable):
    """Convert a variable to a normalized range

    Parameters
    ----------
    variable: scipp.Variable
        The values to be normalized
    minimum: scipp.Variable
        The minimal value that the input `variable` could take
    maximum: scipp.Variable
        The maximal value that the input `variable` could take

    Returns
    -------
    :
        The input `variable` rescaled by the range of allowed values
    """
    full = maximum - minimum
    minimum, full = (in_same_unit(x, to=variable) for x in (minimum, full))
    return (variable - minimum) / full


def is_in_coords(x: DataArray, name: str):
    return name in x.coords or (x.bins is not None and name in x.bins.coords)
