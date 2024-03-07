# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import sciline
import scipp as sc
import scipp.testing
import scippnexus as snx
from scippnexus.application_definitions import nxcansas

from ess.sans.io import save_background_subtracted_iofq
from ess.sans.types import BackgroundSubtractedIofQ, OutFilename, RunNumber, RunTitle


@pytest.mark.parametrize('use_edges', (True, False))
def test_save_background_subtracted_iofq(use_edges, tmp_path):
    def background_subtracted_iofq() -> BackgroundSubtractedIofQ:
        i = sc.arange('Q', 0.0, 400.0)
        i.variances = i.values / 10
        return BackgroundSubtractedIofQ(
            sc.DataArray(
                i,
                coords={
                    'Q': sc.arange('Q', len(i) + int(use_edges), unit='1/angstrom')
                },
            )
        )

    def run_number() -> RunNumber:
        return RunNumber(7419)

    def run_title() -> RunTitle:
        return RunTitle('Test-title')

    out_filename = tmp_path / 'test.nxs'

    providers = (background_subtracted_iofq, run_number, run_title)
    params = {OutFilename: str(out_filename)}
    pipeline = sciline.Pipeline(providers=providers, params=params)
    pipeline.bind_and_call(save_background_subtracted_iofq)

    with snx.File(out_filename, 'r', definitions=nxcansas.definitions) as f:
        loaded = f[()]
    entry = loaded['sasentry']
    assert entry['title'] == 'Test-title'
    assert entry['run'] == 7419

    expected_data = background_subtracted_iofq()
    if use_edges:
        expected_data.coords['Q'] = sc.midpoints(expected_data.coords['Q'])
    sc.testing.assert_identical(entry['sasdata'].coords['Q'], expected_data.coords['Q'])
    # The conversion to stddevs in NeXus loses precision.
    assert sc.allclose(entry['sasdata'].data, expected_data.data)
