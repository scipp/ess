# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pathlib
import subprocess
import time
from contextlib import contextmanager
from enum import Enum

import h5py
import pydantic
import pytest
import scipp as sc
import scippnexus as snx
from scipp.testing import assert_identical

from ess.nmx._executable_helper import (
    InputConfig,
    OutputConfig,
    ReductionConfig,
    WorkflowConfig,
    build_reduction_argument_parser,
    reduction_config_from_args,
)
from ess.nmx.configurations import TimeBinCoordinate, TimeBinUnit, to_command_arguments
from ess.nmx.executables import reduction
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
            assert testing_value != default_value, (
                f"Value for '{key}' is default: {testing_value}"
            )


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
        time_bin_coordinate=TimeBinCoordinate.event_time_offset,
        time_bin_unit=TimeBinUnit.us,
        tof_simulation_num_neutrons=700_000,
        tof_simulation_max_wavelength=5.0,
        tof_simulation_min_wavelength=1.0,
        tof_simulation_min_ltotal=140.0,
        tof_simulation_max_ltotal=200.0,
        tof_simulation_seed=12345,
    )
    output_options = OutputConfig(
        output_file='test-output.h5',
        compression=Compression.NONE,
        verbose=True,
        skip_file_output=True,
        overwrite=True,
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
    assert arg_list == to_command_arguments(config=expected_config, one_line=False)

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


def _check_output_file(output_file_path: pathlib.Path, nbins: int):
    detector_names = [f'detector_panel_{i}' for i in range(3)]
    mandatory_fields = (
        'data',
        'distance',
        'fast_axis',
        'slow_axis',
        'origin',
        'x_pixel_size',
        'y_pixel_size',
        'origin',
    )
    with snx.File(output_file_path, 'r') as f:
        # Test
        assert f['entry/instrument/name'][()] == 'NMX'
        for name in detector_names:
            det_gr = f[f'entry/instrument/{name}']
            assert det_gr is not None
            toa_edges = det_gr['time_of_flight'][()]
            assert len(toa_edges) == nbins
            assert all(field_name in det_gr for field_name in mandatory_fields)


def test_executable_runs(small_nmx_nexus_path, tmp_path: pathlib.Path):
    """Test that the executable runs and returns the expected output."""
    output_file = tmp_path / "output.h5"
    assert not output_file.exists()

    nbins = 20  # Small number of bins for testing.
    # The output has 1280x1280 pixels per detector per time bin.
    commands = (
        'essnmx-reduce',
        '--input-file',
        small_nmx_nexus_path,
        '--nbins',
        str(nbins),
        '--output-file',
        output_file.as_posix(),
    )
    # Validate that all commands are strings and contain no unsafe characters
    result = subprocess.run(  # noqa: S603 - We are not accepting arbitrary input here.
        commands, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0
    assert output_file.exists()
    _check_output_file(output_file, nbins=nbins)


@contextmanager
def known_warnings():
    with pytest.warns(RuntimeWarning, match="No crystal rotation*"):
        yield


@pytest.fixture
def temp_output_file(tmp_path: pathlib.Path):
    output_file_path = tmp_path / "scipp_output.h5"
    yield output_file_path
    if output_file_path.exists():
        output_file_path.unlink()


@pytest.fixture
def reduction_config(
    small_nmx_nexus_path: pathlib.Path, temp_output_file: pathlib.Path
) -> ReductionConfig:
    input_config = InputConfig(input_file=[small_nmx_nexus_path.as_posix()])
    # Compression option is not default (NONE) but
    # the actual default compression option, BITSHUFFLE_LZ4,
    # only properly works in linux so we set it to NONE here
    # for convenience of testing on all platforms.
    output_config = OutputConfig(
        output_file=temp_output_file.as_posix(),
        compression=Compression.NONE,
        skip_file_output=True,  # No need to write output file for most tests.
    )
    return ReductionConfig(inputs=input_config, output=output_config)


def _retrieve_one_hist(results: sc.DataGroup) -> sc.DataArray:
    """Helper to retrieve the first DataArray from the results dictionary."""
    return results['histogram']['detector_panel_0']


def test_reduction_default_settings(reduction_config: ReductionConfig) -> None:
    # Only check if reduction runs without errors with default settings.
    with known_warnings():
        reduction(config=reduction_config)


def test_reduction_only_number_of_time_bins(reduction_config: ReductionConfig) -> None:
    reduction_config.workflow.nbins = 20
    with known_warnings():
        hist = _retrieve_one_hist(reduction(config=reduction_config))

    # Check that the number of time bins is as expected.
    assert len(hist.coords['tof']) == 21  # nbins + 1 edges


def test_histogram_event_time_offset(reduction_config: ReductionConfig) -> None:
    reduction_config.workflow.nbins = 20
    reduction_config.workflow.time_bin_coordinate = TimeBinCoordinate.event_time_offset
    with known_warnings():
        hist = _retrieve_one_hist(reduction(config=reduction_config))

    # Check that the number of time bins is as expected.
    assert len(hist.coords['event_time_offset']) == 21  # nbins + 1 edges
    # Check if the histogram result is reasonable
    zero = sc.scalar(0.0, unit='counts', dtype='float32', variance=0.0)
    assert bool(hist.data.sum() > zero)


def test_histogram_invalid_min_max_raises(reduction_config: ReductionConfig) -> None:
    reduction_config.workflow.min_time_bin = 120
    reduction_config.workflow.max_time_bin = 100
    with pytest.raises(ValueError, match='Cannot build a time bin edges coordinate'):
        with known_warnings():
            reduction(config=reduction_config)


def test_histogram_invalid_min_max_raises_eto(
    reduction_config: ReductionConfig,
) -> None:
    reduction_config.workflow.time_bin_coordinate = TimeBinCoordinate.event_time_offset
    reduction_config.workflow.min_time_bin = 50
    reduction_config.workflow.max_time_bin = 40
    with pytest.raises(ValueError, match='Cannot build a time bin edges coordinate'):
        with known_warnings():
            reduction(config=reduction_config)


@pytest.mark.parametrize(
    argnames="t_coord",
    argvalues=[TimeBinCoordinate.time_of_flight, TimeBinCoordinate.event_time_offset],
)
def test_histogram_out_of_range_min_warns(
    reduction_config: ReductionConfig, t_coord: TimeBinCoordinate
) -> None:
    reduction_config.workflow.time_bin_coordinate = t_coord
    reduction_config.workflow.nbins = 20
    reduction_config.workflow.min_time_bin = 1_000
    reduction_config.workflow.max_time_bin = 2_000
    with pytest.warns(UserWarning, match='is bigger than all'):
        with known_warnings():
            results = reduction(config=reduction_config)

    for da in results['histogram'].values():
        assert_identical(
            da.data.sum(), sc.scalar(0.0, unit='counts', dtype='float32', variance=0.0)
        )


@pytest.mark.parametrize(
    argnames="t_coord",
    argvalues=[TimeBinCoordinate.time_of_flight, TimeBinCoordinate.event_time_offset],
)
def test_histogram_out_of_range_max_warns(
    reduction_config: ReductionConfig, t_coord: TimeBinCoordinate
) -> None:
    reduction_config.workflow.time_bin_coordinate = t_coord
    reduction_config.workflow.nbins = 10
    reduction_config.workflow.min_time_bin = -1
    reduction_config.workflow.max_time_bin = 0
    with pytest.warns(UserWarning, match='is smaller than all'):
        with known_warnings():
            results = reduction(config=reduction_config)

    for da in results['histogram'].values():
        assert_identical(
            da.data.sum(), sc.scalar(0.0, unit='counts', dtype='float32', variance=0.0)
        )


@pytest.fixture
def tof_lut_file_path(tmp_path: pathlib.Path):
    """Fixture to provide the path to the small NMX NeXus file."""
    from dataclasses import is_dataclass

    from ess.reduce.time_of_flight import TimeOfFlightLookupTable

    from ess.nmx.workflows import initialize_nmx_workflow

    # Simply use the default workflow for testing.
    workflow = initialize_nmx_workflow(config=WorkflowConfig())
    tof_lut: TimeOfFlightLookupTable = workflow.compute(TimeOfFlightLookupTable)

    # Change the tof range a bit for testing.
    if isinstance(tof_lut, sc.DataArray):
        tof_lut *= 2
    elif is_dataclass(tof_lut):
        tof_lut.array *= 2
    else:
        raise TypeError("Unexpected type for TOF lookup table.")

    lut_file_path = tmp_path / "nmx_tof_lookup_table.h5"
    tof_lut.save_hdf5(lut_file_path.as_posix())
    yield lut_file_path
    if lut_file_path.exists():
        lut_file_path.unlink()


def test_reduction_with_tof_lut_file(
    reduction_config: ReductionConfig, tof_lut_file_path: pathlib.Path
) -> None:
    # Make sure the config uses no TOF lookup table file initially.
    assert reduction_config.workflow.tof_lookup_table_file_path is None
    with known_warnings():
        default_results = reduction(config=reduction_config)

    # Update config to use the TOF lookup table file.
    reduction_config.workflow.tof_lookup_table_file_path = tof_lut_file_path.as_posix()
    with known_warnings():
        results = reduction(config=reduction_config)

    for default_hist, hist in zip(
        default_results['histogram'].values(),
        results['histogram'].values(),
        strict=True,
    ):
        tof_edges_default = default_hist.coords['tof']
        tof_edges = hist.coords['tof']
        assert_identical(default_hist.data, hist.data)
        assert_identical(tof_edges_default * 2, tof_edges)


def test_reduction_succeed_when_skipping_evenif_output_file_exists(
    reduction_config: ReductionConfig, temp_output_file: pathlib.Path
) -> None:
    # Make sure the file exists
    temp_output_file.touch(exist_ok=True)
    # Make sure the file output is skipped.
    reduction_config.output.skip_file_output = True

    # Adjust workflow setting to finish fast.
    reduction_config.workflow.nbins = 2
    reduction_config.workflow.time_bin_coordinate = TimeBinCoordinate.event_time_offset
    with known_warnings():
        reduction(config=reduction_config)


def test_reduction_fails_fast_if_output_file_exists(
    reduction_config: ReductionConfig, temp_output_file: pathlib.Path
) -> None:
    # Make sure the file exists
    temp_output_file.touch()
    # Make sure file output is NOT skipped.
    reduction_config.output.skip_file_output = False

    start = time.time()
    with pytest.raises(FileExistsError):
        reduction(config=reduction_config)
    finish = time.time()

    # Check if the `reduction` call fails within 1 second.
    # There is no special reason why it is 1 second.
    # It should just fail as fast as possible.
    assert finish - start < 1


def test_reduction_compression_gzip(
    reduction_config: ReductionConfig, tmp_path: pathlib.Path
) -> None:
    reduction_config.output.skip_file_output = False
    reduction_config.workflow.nbins = 5  # For faster test
    file_paths: dict[Compression, pathlib.Path] = {}

    for compress_mode in (Compression.NONE, Compression.GZIP):
        reduction_config.output.compression = compress_mode
        cur_file_path = tmp_path / f'compress_{compress_mode}_output.hdf'
        file_paths[compress_mode] = cur_file_path
        assert not cur_file_path.exists()
        reduction_config.output.output_file = cur_file_path.as_posix()
        # Running the whole reduction instead of only saving the file on purpose.
        with known_warnings():
            reduction(config=reduction_config)
        assert cur_file_path.exists()

    assert (
        file_paths[Compression.NONE].stat().st_size
        > file_paths[Compression.GZIP].stat().st_size
    )
    with h5py.File(file_paths[Compression.NONE]) as file:
        for i in range(3):
            assert file[f'entry/instrument/detector_panel_{i}/data'].chunks is None

    with h5py.File(file_paths[Compression.GZIP]) as file:
        for i in range(3):
            data_path = f'entry/instrument/detector_panel_{i}/data'
            assert file[data_path].chunks == (1280, 1280, 1)
            assert file[data_path].compression == 'gzip'
            assert file[data_path].compression_opts == 4


try:
    # Just checking availability
    import bitshuffle.h5  # noqa: F401
except ImportError:
    BITSHUFFLE_AVAILABLE = False
else:
    BITSHUFFLE_AVAILABLE = True


@pytest.mark.skipif(
    not BITSHUFFLE_AVAILABLE,
    reason="Bitshuffle is not available in this environment.",
)
def test_reduction_compression_bitshuffle_smaller_than_gzip(
    reduction_config: ReductionConfig, tmp_path: pathlib.Path
) -> None:
    reduction_config.output.skip_file_output = False
    reduction_config.workflow.nbins = 5  # For faster test
    file_paths: dict[Compression, pathlib.Path] = {}
    total_times: dict[Compression, pathlib.Path] = {}

    for compress_mode in (Compression.GZIP, Compression.BITSHUFFLE_LZ4):
        reduction_config.output.compression = compress_mode
        cur_file_path = tmp_path / f'compress_{compress_mode}_output.hdf'
        file_paths[compress_mode] = cur_file_path
        assert not cur_file_path.exists()
        reduction_config.output.output_file = cur_file_path.as_posix()
        # Running the whole reduction instead of only saving the file on purpose.
        with known_warnings():
            start = time.time()
            reduction(config=reduction_config)
            end = time.time()

        assert cur_file_path.exists()
        total_times[compress_mode] = end - start

    # GZIP is expected to have better compression ratio than BITSHUFFLE
    assert (
        file_paths[Compression.BITSHUFFLE_LZ4].stat().st_size
        > file_paths[Compression.GZIP].stat().st_size
    )
    # BITSHUFFLE is expected to be faster than GZIP
    assert total_times[Compression.BITSHUFFLE_LZ4] < total_times[Compression.GZIP]

    with h5py.File(file_paths[Compression.GZIP]) as file:
        for i in range(3):
            data_path = f'entry/instrument/detector_panel_{i}/data'
            assert file[data_path].chunks == (1280, 1280, 1)
            assert file[data_path].compression == 'gzip'

    with h5py.File(file_paths[Compression.BITSHUFFLE_LZ4]) as file:
        for i in range(3):
            data_path = f'entry/instrument/detector_panel_{i}/data'
            assert file[data_path].chunks == (1280, 1280, 1)
            # For some reason it doesn't write the compression.
            # so we check the filter instead.
            # assert file[data_path].compression == 'bitshuffle'
            assert '32008' in file[data_path]._filters


@pytest.mark.skipif(
    BITSHUFFLE_AVAILABLE,
    reason="Bitshuffle is available in this environment so it won't fall back.",
)
def test_reduction_compression_bitshuffle_fall_back_to_gzip(
    reduction_config: ReductionConfig, temp_output_file: pathlib.Path
) -> None:
    reduction_config.output.skip_file_output = False
    reduction_config.workflow.nbins = 5  # For faster test
    reduction_config.output.compression = Compression.BITSHUFFLE_LZ4
    reduction_config.output.output_file = temp_output_file.as_posix()

    with known_warnings():
        with pytest.warns(UserWarning, match='bitshuffle.h5'):
            reduction(config=reduction_config)

    with h5py.File(temp_output_file) as file:
        for i in range(3):
            data_path = f'entry/instrument/detector_panel_{i}/data'
            assert file[data_path].chunks == (1280, 1280, 1)
            assert file[data_path].compression == 'gzip'


def test_reduction_duplicated_path_raises(reduction_config: ReductionConfig) -> None:
    # Run with two files with same names.
    reduction_config.inputs.input_file = reduction_config.inputs.input_file * 2
    with pytest.raises(
        ValueError, match=r'Duplicated file paths or pattern found.*small_nmx_nexus.hdf'
    ):
        reduction(config=reduction_config)
