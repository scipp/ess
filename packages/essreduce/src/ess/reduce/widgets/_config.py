# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from ipywidgets import Layout


def full_width_layout(**kwargs) -> Layout:
    kwargs.setdefault('width', '100%')
    return Layout(**kwargs)


default_layout = full_width_layout()
default_style = {'description_width': 'auto'}
