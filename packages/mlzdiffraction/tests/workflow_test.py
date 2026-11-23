# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import sciline
from mlz.diffraction.powder import PowderReduceP2D
from mlz.diffraction.texture import TextureReduce


def test_powder_reduce_p2d_constructs_pipeline() -> None:
    assert isinstance(PowderReduceP2D(), sciline.Pipeline)


def test_texture_reduce_constructs_pipeline() -> None:
    assert isinstance(TextureReduce(), sciline.Pipeline)
