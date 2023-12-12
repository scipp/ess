# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from .detector import default_params as detector_default_params
from .loader import load_nmx_file
from .reduction import get_grouped_by_pixel_id

providers = (load_nmx_file, get_grouped_by_pixel_id)


def collect_default_parameters() -> dict:
    return dict(detector_default_params)
