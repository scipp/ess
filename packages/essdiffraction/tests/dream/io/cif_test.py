# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import io

import pytest
import scipp as sc
from scippneutron.io import cif

import ess.dream.io.cif
from ess.powder.types import CIFAuthors, IofDspacing


@pytest.fixture
def iofd() -> IofDspacing:
    return IofDspacing(
        sc.DataArray(
            sc.array(dims=['dspacing'], values=[2.1, 3.2], variances=[0.3, 0.4]),
            coords={'dspacing': sc.linspace('dspacing', 0.1, 1.2, 3, unit='angstrom')},
        )
    )


def save_reduced_dspacing_to_str(cif_: cif.CIF) -> str:
    buffer = io.StringIO()
    cif_.save(buffer)
    buffer.seek(0)
    return buffer.read()


def test_save_reduced_dspacing(iofd: IofDspacing) -> None:
    from ess.dream import __version__

    author = cif.Author(name='John Doe')
    cif_ = ess.dream.io.cif.prepare_reduced_dspacing_cif(
        iofd, authors=CIFAuthors([author])
    )
    result = save_reduced_dspacing_to_str(cif_)

    assert "_audit_contact_author.name 'John Doe'" in result
    assert f"_computing.diffrn_reduction 'ess.dream v{__version__}'" in result
    assert '_diffrn_source.beamline DREAM' in result

    loop_header = """loop_
_pd_proc.d_spacing
_pd_proc.intensity_net
_pd_proc.intensity_net_su
"""
    assert loop_header in result
