# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pathlib
import subprocess
from enum import Enum

import pydantic
import pytest
import scipp as sc
import scippnexus as snx
from scipp.testing import assert_allclose

from ess.nmx._executable_helper import (
    InputConfig,
    OutputConfig,
    ReductionConfig,
    TimeBinCoordinate,
    TimeBinUnit,
    WorkflowConfig,
    build_reduction_argument_parser,
    reduction_config_from_args,
    to_command_arguments,
)
from ess.nmx.types import Compression


def _build_arg_list_from_pydantic_instance(*instances: pydantic.BaseModel) -> list[str]:
    args = {}
    for instance in instances:
        args.update(instance.model_dump(mode='python'))
    args = {f"--{k.replace('_', '-')}": v for k, v in args.items() if v is not None}

    arg_list = []
    for k, v in args.items():
        if not isinstance(v, bool):
            arg_list.append(k)
            if isinstance(v, list):
                arg_list.extend(str(item) for item in v)
            elif isinstance(v, Enum):
                arg_list.append(v.name)
            else:
                arg_list.append(str(v))
        elif v is True:
            arg_list.append(k)

    return arg_list


def _default_config() -> ReductionConfig:
    """Helper to create a default ReductionConfig instance."""
    return ReductionConfig(
        inputs=InputConfig(input_file=['']),
        workflow=WorkflowConfig(),
        output=OutputConfig(),
    )


def _check_non_default_config(testing_config: ReductionConfig) -> None:
    """Helper to check that all values in the config are non-default."""
    default_config = _default_config()
    testing_children = testing_config._children
    default_children = default_config._children
    for testing_child, default_child in zip(
        testing_children, default_children, strict=True
    ):
        testing_model = testing_child.model_dump(mode='python')
        default_model = default_child.model_dump(mode='python')
        for key, testing_value in testing_model.items():
            if key == 'tof_lookup_table_file_path':
                # This value may be None or default, so we skip the check.
                continue
            default_value = default_model[key]
            assert (
                testing_value != default_value
            ), f"Value for '{key}' is default: {testing_value}"


def test_reduction_config() -> None:
    """Test ReductionConfig argument parsing."""
    # Build config instances with non-default values.
    input_options = InputConfig(
        input_file=['test-input.h5'],
        swmr=True,
        detector_ids=[0, 1, 2, 3],
        iter_chunk=True,
        chunk_size_pulse=10,
        chunk_size_events=100000,
    )
    workflow_options = WorkflowConfig(
        nbins=100,
        min_time_bin=10,
        max_time_bin=100_000,
        time_bin_coordinate=TimeBinCoordinate.time_of_flight,
        time_bin_unit=TimeBinUnit.us,
        tof_simulation_max_wavelength=5.0,
        tof_simulation_min_wavelength=1.0,
        tof_simulation_seed=12345,
    )
    output_options = OutputConfig(
        output_file='test-output.h5', compression=Compression.NONE, verbose=True
    )
    expected_config = ReductionConfig(
        inputs=input_options, workflow=workflow_options, output=output_options
    )
    # Check if all values are non-default.
    _check_non_default_config(expected_config)

    # Build argument list manually, not using `to_command_arguments` to test it.
    arg_list = _build_arg_list_from_pydantic_instance(
        input_options, workflow_options, output_options
    )
    assert arg_list == to_command_arguments(expected_config, one_line=False)

    # Parse arguments and build config from them.
    parser = build_reduction_argument_parser()
    args = parser.parse_args(arg_list)
    config = reduction_config_from_args(args)
    assert expected_config == config


@pytest.fixture(scope="session")
def small_nmx_nexus_path():
    """Fixture to provide the path to the small NMX NeXus file."""
    from ess.nmx.data import get_small_nmx_nexus

    return get_small_nmx_nexus()


def _check_output_file(
    output_file_path: pathlib.Path, expected_toa_output: sc.Variable
):
    detector_names = [f'detector_panel_{i}' for i in range(3)]
    with snx.File(output_file_path, 'r') as f:
        # Test
        for name in detector_names:
            det_gr = f[f'entry/instrument/{name}']
            assert det_gr is not None
            toa_edges = det_gr['time_of_flight'][()]
            assert_allclose(toa_edges, expected_toa_output)


def test_executable_runs(small_nmx_nexus_path, tmp_path: pathlib.Path):
    """Test that the executable runs and returns the expected output."""
    output_file = tmp_path / "output.h5"
    assert not output_file.exists()

    nbins = 20  # Small number of bins for testing.
    # The output has 1280x1280 pixels per detector per time bin.
    expected_toa_bins = sc.linspace(
        dim='dim_0',
        start=2,  # Unrealistic number for testing
        stop=int((1 / 15) * 1_000),  # Unrealistic number for testing
        num=nbins + 1,
        unit='ms',
    )
    expected_toa_output = sc.midpoints(expected_toa_bins, dim='dim_0').to(unit='ns')

    commands = (
        'essnmx-reduce',
        '--input-file',
        small_nmx_nexus_path,
        '--nbins',
        str(nbins),
        '--output-file',
        output_file.as_posix(),
        '--min-time-bin',
        str(int(expected_toa_bins.min().value)),
        '--max-time-bin',
        str(int(expected_toa_bins.max().value)),
    )
    # Validate that all commands are strings and contain no unsafe characters
    result = subprocess.run(  # noqa: S603 - We are not accepting arbitrary input here.
        commands,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert output_file.exists()
    _check_output_file(output_file, expected_toa_output=expected_toa_output)
