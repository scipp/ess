# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from dataclasses import fields, is_dataclass

import scipp as sc


def _is_nested(obj) -> bool:
    return is_dataclass(obj) or isinstance(obj, sc.DataGroup | dict)


def to_datagroup(obj, *, drop_nones: bool = True) -> sc.DataGroup:
    if is_dataclass(obj):
        return sc.DataGroup(
            {
                field.name: to_datagroup(value, drop_nones=drop_nones)
                if _is_nested(value)
                else value
                for field in fields(obj)
                if (value := getattr(obj, field.name)) is not None and drop_nones
            }
        )
    elif isinstance(obj, sc.DataGroup | dict):
        return sc.DataGroup(
            {
                name: to_datagroup(value, drop_nones=drop_nones)
                if _is_nested(value)
                else value
                for name, value in obj.items()
                if value is not None and drop_nones
            }
        )
    else:
        return obj
