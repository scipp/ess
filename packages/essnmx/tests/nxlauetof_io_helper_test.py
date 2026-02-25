# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import pathlib
from contextlib import contextmanager

import pytest
from scipp.testing.assertions import assert_allclose, assert_identical

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

    # Adjust original result to be same as expected loaded data group.
    original_result_dg.pop('lookup_table')
    original_positions = {}
    detectors = original_result_dg['instrument']['detectors']
    for det_name, det in detectors.items():
        # Removing coordinates that are not kept in the file or reconstructed.
        det['data'].coords.pop('Ltotal')
        det['data'].coords.pop('detector_number')
        det['data'].coords.pop('x_pixel_offset')
        det['data'].coords.pop('y_pixel_offset')
        # Saving position coordinate to compare them by allclose instead of eq.
        original_positions[det_name] = det['data'].coords.pop('position')

    with pytest.warns(UserWarning, match=r'Could not determine'):
        loaded_dg = load_essnmx_nxlauetof(reduction_config.output.output_file)

    loaded_detector_positions = {}
    for det_name, loaded_det in loaded_dg['instrument']['detectors'].items():
        loaded_detector_positions[det_name] = loaded_det['data'].coords.pop('position')

    assert_identical(loaded_dg, original_result_dg)
    # Using the x_pixel_size of the first panel to get absolute tolerance.
    pixel_size = next(iter(detectors.values()))['metadata']['x_pixel_size']
    atol = pixel_size / 10.0
    for det_name, original_position in original_positions.items():
        loaded_position = loaded_detector_positions[det_name]
        assert_allclose(original_position, loaded_position, atol=atol)
