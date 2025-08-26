# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# flake8: noqa: F403, F405

import scipp as sc
import scipp.constants
import scipp.testing

from ess.amor.utils import qgrid


def test_qgrid_provider():
    grid = qgrid(
        detector_rotation=sc.scalar(2, unit='deg'),
        sample_rotation=sc.scalar(0.9, unit='deg'),
        wbins=sc.linspace('wavelength', 3, 12, 10, unit='angstrom'),
        bdlims=(sc.scalar(-0.75, unit='deg'), sc.scalar(0.75, unit='deg')),
    )
    sc.testing.assert_allclose(
        grid[0],
        4
        * sc.constants.pi
        * sc.sin(
            sc.scalar(-0.75, unit='deg').to(unit='rad')
            + sc.scalar(1.1, unit='deg').to(unit='rad')
        )
        / sc.scalar(12, unit='angstrom'),
    )
    sc.testing.assert_allclose(
        grid[-1],
        4
        * sc.constants.pi
        * sc.sin(
            sc.scalar(0.75, unit='deg').to(unit='rad')
            + sc.scalar(1.1, unit='deg').to(unit='rad')
        )
        / sc.scalar(3, unit='angstrom'),
    )


def test_qgrid_provider_minimum_q():
    grid = qgrid(
        detector_rotation=sc.scalar(1.2, unit='deg'),
        sample_rotation=sc.scalar(0.7, unit='deg'),
        wbins=sc.linspace('wavelength', 3, 12, 10, unit='angstrom'),
        bdlims=(sc.scalar(-0.75, unit='deg'), sc.scalar(0.75, unit='deg')),
    )
    sc.testing.assert_allclose(grid[0], sc.scalar(0.001, unit='1/angstrom'))
