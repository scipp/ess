# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pathlib

import pandas as pd
import pytest
import sciline as sl
import scipp as sc

from ess.nmx import NMXMcStasWorkflow
from ess.nmx.data import get_small_mcstas
from ess.nmx.mcstas.reduction import merge_panels
from ess.nmx.mcstas.types import (
    DetectorIndex,
    FilePath,
    MaximumCounts,
    NMXRawEventCountsDataGroup,
    NMXReducedDataGroup,
    TimeBinSteps,
)


@pytest.fixture(params=[get_small_mcstas])
def mcstas_file_path(request: pytest.FixtureRequest) -> pathlib.Path:
    return request.param()


@pytest.fixture
def mcstas_workflow(mcstas_file_path: pathlib.Path) -> sl.Pipeline:
    wf = NMXMcStasWorkflow()
    wf[FilePath] = mcstas_file_path
    wf[TimeBinSteps] = 50
    return wf


@pytest.fixture
def multi_bank_mcstas_workflow(mcstas_workflow: sl.Pipeline) -> sl.Pipeline:
    pl = mcstas_workflow.copy()
    pl[NMXReducedDataGroup] = (
        pl[NMXReducedDataGroup]
        .map(pd.DataFrame({DetectorIndex: range(3)}).rename_axis('panel'))
        .reduce(index='panel', func=merge_panels)
    )
    return pl


def test_pipeline_builder(
    mcstas_workflow: sl.Pipeline, mcstas_file_path: pathlib.Path
) -> None:
    assert mcstas_workflow.get(FilePath).compute() == mcstas_file_path


def test_pipeline_mcstas_loader(mcstas_workflow: sl.Pipeline) -> None:
    """Test if the loader graph is complete."""
    mcstas_workflow[DetectorIndex] = 0
    nmx_data = mcstas_workflow.compute(NMXRawEventCountsDataGroup)
    assert isinstance(nmx_data, sc.DataGroup)
    assert nmx_data.sizes['id'] == 1280 * 1280


def test_pipeline_mcstas_reduction(multi_bank_mcstas_workflow: sl.Pipeline) -> None:
    """Test if the loader graph is complete."""
    from scipp.testing import assert_allclose, assert_identical

    from ess.nmx.mcstas import default_parameters

    nmx_reduced_data = multi_bank_mcstas_workflow.compute(NMXReducedDataGroup)
    assert nmx_reduced_data.shape == (3, (1280, 1280)[0] * (1280, 1280)[1], 50)
    # Panel, Pixels, Time bins
    assert isinstance(nmx_reduced_data, sc.DataGroup)

    # Check maximum value of weights.
    assert_allclose(
        nmx_reduced_data['counts'].max().data,
        sc.scalar(default_parameters[MaximumCounts], unit='counts', dtype=float),
        atol=sc.scalar(1e-10, unit='counts'),
        rtol=sc.scalar(1e-8),
    )
    assert_identical(
        nmx_reduced_data['proton_charge'],
        sc.scalar(1e-4, unit='dimensionless')
        * nmx_reduced_data['counts'].data.sum('id').sum('t'),
    )
    assert nmx_reduced_data.sizes['t'] == 50
