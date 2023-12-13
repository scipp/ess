# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from .detector import default_params as detector_default_params


def collect_default_parameters() -> dict:
    return dict(detector_default_params)
