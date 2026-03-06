# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from dataclasses import fields, is_dataclass

import scipp as sc


def _is_nested(obj) -> bool:
    return is_dataclass(obj) or isinstance(obj, sc.DataGroup | dict)


def to_datagroup(obj) -> sc.DataGroup:
    if is_dataclass(obj):
        return sc.DataGroup(
            {
                field.name: to_datagroup(value)
                if _is_nested(value := getattr(obj, field.name))
                else value
                for field in fields(obj)
            }
        )
    elif isinstance(obj, sc.DataGroup | dict):
        return sc.DataGroup(
            {
                name: to_datagroup(value) if _is_nested(value) else value
                for name, value in obj.items()
            }
        )
    else:
        return obj
