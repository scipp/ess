# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc

from ess.sans.types import QBins

Q = sc.linspace(dim='Q', start=0.1, stop=1.0, num=10, unit='1/angstrom')
Qx = sc.linspace(dim='Qx', start=0.1, stop=1.0, num=10, unit='1/angstrom')
Qy = sc.linspace(dim='Qy', start=0.1, stop=1.0, num=10, unit='1/angstrom')


def test_raises_if_no_args() -> None:
    with pytest.raises(ValueError):
        QBins()


def test_raises_if_Q_and_Qx_or_Qy_given() -> None:
    with pytest.raises(ValueError):
        QBins(Q, Qx=Qx)
    with pytest.raises(ValueError):
        QBins(Q, Qy=Qy)
    with pytest.raises(ValueError):
        QBins(Q, Qx=Qx, Qy=Qy)


def test_create_1d_sets_edges_property() -> None:
    qbins = QBins(Q)
    assert qbins.edges == {'Q': Q}
    assert qbins.dims == ('Q',)


def test_create_2d_sets_edges_property_in_y_x_order() -> None:
    qbins = QBins(Qx=Qx, Qy=Qy)
    assert qbins.edges == {'Qy': Qy, 'Qx': Qx}
    assert qbins.dims == ('Qy', 'Qx')
