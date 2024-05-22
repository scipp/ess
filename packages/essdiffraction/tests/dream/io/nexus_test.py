# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import sciline
from ess import dream
from ess.dream import nexus
from ess.powder.types import (
    Filename,
    NeXusDetectorName,
    RawDetectorData,
    ReducibleDetectorData,
    SampleRun,
)

bank_dims = {'wire', 'module', 'segment', 'strip', 'counter'}
hr_sans_dims = {'strip', 'other'}


@pytest.fixture()
def providers():
    return (
        nexus.dummy_load_sample,
        nexus.load_nexus_source,
        nexus.load_nexus_detector,
        nexus.get_sample_position,
        nexus.get_source_position,
        nexus.get_detector_data,
        nexus.patch_detector_data,
    )


@pytest.fixture(
    params=[
        'mantle_detector',
        'endcap_backward_detector',
        'endcap_forward_detector',
        'high_resolution_detector',
        # TODO: the 'sans_detector' is strange in the current files
    ]
)
def params(request):
    params = {
        Filename[SampleRun]: dream.data.get_path('DREAM_nexus_sorted-2023-12-07.nxs'),
        NeXusDetectorName: request.param,
    }
    return params


def test_can_load_nexus_detector_data(providers, params):
    pipeline = sciline.Pipeline(params=params, providers=providers)
    result = pipeline.compute(RawDetectorData[SampleRun])
    assert (
        set(result.dims) == hr_sans_dims
        if params[NeXusDetectorName]
        in (
            'high_resolution_detector',
            'sans_detector',
        )
        else bank_dims
    )


def test_load_fails_with_bad_detector_name(providers):
    params = {
        Filename[SampleRun]: dream.data.get_path('DREAM_nexus_sorted-2023-12-07.nxs'),
        NeXusDetectorName: 'bad_detector',
    }
    pipeline = sciline.Pipeline(params=params, providers=providers)
    with pytest.raises(KeyError, match='bad_detector'):
        pipeline.compute(RawDetectorData[SampleRun])


def test_patch_nexus_detector_data(providers, params):
    pipeline = sciline.Pipeline(params=params, providers=providers)
    result = pipeline.compute(ReducibleDetectorData[SampleRun])
    assert (
        set(result.dims) == hr_sans_dims
        if params[NeXusDetectorName]
        in (
            'high_resolution_detector',
            'sans_detector',
        )
        else bank_dims
    )
    assert "source_position" in result.coords
    assert "sample_position" in result.coords
