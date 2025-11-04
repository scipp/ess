# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pathlib
import subprocess

import pytest
import scipp as sc
import scippnexus as snx
from scipp.testing import assert_allclose


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
        '--input_file',
        small_nmx_nexus_path,
        '--nbins',
        str(nbins),
        '--output_file',
        output_file.as_posix(),
        '--min-toa',
        str(int(expected_toa_bins.min().value)),
        '--max-toa',
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
