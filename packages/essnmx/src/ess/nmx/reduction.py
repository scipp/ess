# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType

import scipp as sc

from .detector import NumberOfDetectors
from .loader import NMXData

GroupedByPixelID = NewType("GroupedByPixelID", sc.DataArray)


def get_grouped_by_pixel_id(
    loaded: NMXData, num_panels: NumberOfDetectors
) -> GroupedByPixelID:
    """group events by pixel ID"""
    grouped = loaded.events.group(loaded.all_pixel_ids)

    return GroupedByPixelID(
        grouped.fold(dim='id', sizes={'panel': num_panels, 'id': -1})
    )
