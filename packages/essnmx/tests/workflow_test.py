# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import sciline as sl
import scipp as sc

from ess.nmx.data import small_mcstas_2_sample, small_mcstas_3_sample


@pytest.fixture(params=[small_mcstas_2_sample, small_mcstas_3_sample])
def mcstas_file_path(
    request: pytest.FixtureRequest, mcstas_2_deprecation_warning_context
) -> str:
    if request.param == small_mcstas_2_sample:
        with mcstas_2_deprecation_warning_context():
            return request.param()

    return request.param()


@pytest.fixture
def mcstas_workflow(mcstas_file_path: str) -> sl.Pipeline:
    from ess.nmx.mcstas_loader import (
        DefaultMaximumProbability,
        DetectorBankName,
        EventWeightsConverter,
        InputFilepath,
        MaximumProbability,
        ProtonChargeConverter,
        event_weights_from_probability,
        load_mcstas_nexus,
        proton_charge_from_event_data,
    )
    from ess.nmx.reduction import TimeBinSteps, bin_time_of_arrival

    return sl.Pipeline(
        [load_mcstas_nexus, bin_time_of_arrival],
        params={
            InputFilepath: mcstas_file_path,
            MaximumProbability: DefaultMaximumProbability,
            TimeBinSteps: TimeBinSteps(50),
            EventWeightsConverter: event_weights_from_probability,
            ProtonChargeConverter: proton_charge_from_event_data,
            DetectorBankName: 'bank01',
        },
    )


def test_pipeline_builder(mcstas_workflow: sl.Pipeline, mcstas_file_path: str) -> None:
    from ess.nmx.mcstas_loader import InputFilepath

    assert mcstas_workflow.get(InputFilepath).compute() == mcstas_file_path


def test_pipeline_mcstas_loader(mcstas_workflow: sl.Pipeline) -> None:
    """Test if the loader graph is complete."""
    from ess.nmx.mcstas_loader import NMXData

    nmx_data = mcstas_workflow.compute(NMXData)
    assert isinstance(nmx_data, sc.DataGroup)
    assert nmx_data.sizes['id'] == 1280 * 1280


def test_pipeline_mcstas_reduction(mcstas_workflow: sl.Pipeline) -> None:
    """Test if the loader graph is complete."""
    from ess.nmx.reduction import NMXReducedData

    nmx_reduced_data = mcstas_workflow.compute(NMXReducedData)
    assert isinstance(nmx_reduced_data, sc.DataGroup)
    assert nmx_reduced_data.sizes['t'] == 50
