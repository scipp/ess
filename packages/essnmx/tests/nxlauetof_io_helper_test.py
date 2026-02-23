# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pathlib
from contextlib import contextmanager

import pytest
from scipp.testing.assertions import assert_identical

from ess.nmx._nxlauetof_io import load_essnmx_nxlauetof
from ess.nmx.configurations import InputConfig, OutputConfig, ReductionConfig
from ess.nmx.executables import reduction
from ess.nmx.types import Compression


@pytest.fixture
def temp_output_file(tmp_path: pathlib.Path):
    output_file_path = tmp_path / "scipp_output.h5"
    yield output_file_path
    if output_file_path.exists():
        output_file_path.unlink()


@pytest.fixture
def reduction_config(temp_output_file: pathlib.Path) -> ReductionConfig:
    from ess.nmx.data import get_small_nmx_nexus

    input_config = InputConfig(input_file=[get_small_nmx_nexus().as_posix()])
    output_config = OutputConfig(
        output_file=temp_output_file.as_posix(),
        compression=Compression.NONE,
        skip_file_output=False,
    )
    return ReductionConfig(inputs=input_config, output=output_config)


@contextmanager
def known_warnings():
    with pytest.warns(RuntimeWarning, match="No crystal rotation*"):
        yield


def test_loaded_data_same_as_in_memory_result(
    reduction_config: ReductionConfig,
) -> None:
    with known_warnings():
        result = reduction(config=reduction_config)
    original_result_dg = result.to_datagroup()

    with pytest.warns(UserWarning, match=r'Could not determine'):
        loaded_dg = load_essnmx_nxlauetof(reduction_config.output.output_file)

    assert_identical(loaded_dg['sample'], original_result_dg['sample'])
