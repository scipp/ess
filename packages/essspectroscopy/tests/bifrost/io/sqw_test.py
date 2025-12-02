# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

# Writing an SQW file is fairly slow.
# So the tests in this module use module-scoped fixtures to write a single file
# that is shared between all tests.
# Function-scoped fixtures allow accessing that file for reading.

import itertools
from collections.abc import Generator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import sciline
import scipp as sc
import scipp.testing
import scippnexus as snx
from scippneutron.io import sqw

from ess import bifrost
from ess.bifrost.data import (
    simulated_elastic_incoherent_with_phonon,
    tof_lookup_table_simulation,
)
from ess.spectroscopy.types import (
    EnergyBins,
    Filename,
    NeXusDetectorName,
    OutFilename,
    PreopenNeXusFile,
    SampleRun,
    SQWBinSizes,
    TimeOfFlightLookupTableFilename,
)

N_DETECTORS = 3
N_ANGLES = 180

BIN_SIZES = {'u1': 6, 'u2': 7, 'u3': 8, 'u4': 9}
ENERGY_BIN_SIZE = 13

# Q projections
U = sc.vector([1, 0, 0], unit="1/angstrom")
V = sc.vector([0, 1, 0], unit="1/angstrom")
W = sc.vector([0, 0, 1], unit="1/angstrom")

N_PIXELS_PER_DETECTOR = 300  # Fixed, not a parameter!


@pytest.fixture(scope='module')
def detector_names() -> list[NeXusDetectorName]:
    with snx.File(simulated_elastic_incoherent_with_phonon()) as f:
        detector_names = list(f['entry/instrument'][snx.NXdetector])
    return detector_names[:N_DETECTORS]


@pytest.fixture(scope='module')
def sample() -> sqw.SqwIXSample:
    return sqw.SqwIXSample(
        name="Vibranium",
        lattice_spacing=sc.vector([2.86, 2.86, 2.86], unit="angstrom"),
        lattice_angle=sc.vector([90.0, 90.0, 90.0], unit="deg"),
    )


@pytest.fixture(scope='module')
def common_workflow(
    detector_names: list[NeXusDetectorName], sample: sqw.SqwIXSample
) -> sciline.Pipeline:
    wf = bifrost.BifrostSimulationWorkflow(detector_names)

    wf[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
    wf[TimeOfFlightLookupTableFilename] = tof_lookup_table_simulation()
    wf[PreopenNeXusFile] = PreopenNeXusFile(True)
    wf[sqw.SqwIXSample] = sample
    wf[EnergyBins] = ENERGY_BIN_SIZE
    wf[SQWBinSizes] = BIN_SIZES

    return wf


@pytest.fixture(scope='module')
def write_file(
    common_workflow: sciline.Pipeline, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    path = tmp_path_factory.mktemp("bifrost-sqw").joinpath("bifrost-simulated.sqw")
    workflow = common_workflow.copy()
    workflow[OutFilename] = path
    workflow.bind_and_call(bifrost.io.sqw.save_sqw)
    return path


@pytest.fixture
def output_file(write_file: Path) -> Generator[sqw.Sqw, None, None]:
    with sqw.Sqw.open(write_file) as file:
        yield file


def test_save_sqw_writes_instrument_metadata(output_file: sqw.Sqw) -> None:
    instruments = output_file.read_data_block("experiment_info", "instruments")

    assert len(instruments) == N_ANGLES
    # All instruments are the same:
    for instrument in instruments[1:]:
        sc.testing.assert_identical(instrument, instruments[0])

    instrument = instruments[0]
    assert instrument.name == "BIFROST"
    sc.testing.assert_identical(instrument.source.frequency, sc.scalar(14.0, unit="Hz"))


def test_save_sqw_writes_sample_metadata(
    output_file: sqw.Sqw, sample: sqw.SqwIXSample
) -> None:
    samples = output_file.read_data_block("experiment_info", "samples")

    assert len(samples) == N_ANGLES
    # All samples are the same:
    for s in samples:
        sc.testing.assert_identical(s, sample)


def test_save_sqw_writes_experiment_metadata(output_file: sqw.Sqw) -> None:
    experiments = output_file.read_data_block("experiment_info", "expdata")

    assert len(experiments) == N_ANGLES
    # N unique run ids
    assert len({experiment.run_id for experiment in experiments}) == N_ANGLES
    for experiment in experiments:
        assert experiment.emode == sqw.EnergyMode.indirect
        assert experiment.u == U
        assert experiment.v == V


def test_save_sqw_writes_dnd_metadata(
    output_file: sqw.Sqw, sample: sqw.SqwIXSample
) -> None:
    metadata = output_file.read_data_block("data", "metadata")

    np.testing.assert_array_equal(
        metadata.axes.n_bins_all_dims.values, list(BIN_SIZES.values())
    )

    sc.testing.assert_identical(metadata.proj.u, U)
    sc.testing.assert_identical(metadata.proj.v, V)
    sc.testing.assert_identical(metadata.proj.w, W)
    assert metadata.proj.type == "aaa"
    assert not metadata.proj.non_orthogonal

    sc.testing.assert_identical(metadata.proj.lattice_spacing, sample.lattice_spacing)
    sc.testing.assert_identical(metadata.proj.lattice_angle, sample.lattice_angle)


def test_save_sqw_writes_dnd_data(output_file: sqw.Sqw) -> None:
    dnd = output_file.read_data_block("data", "nd_data")
    values, errors, counts = dnd
    # Fortran order
    expected_shape = tuple(BIN_SIZES.values())[::-1]
    assert values.shape == expected_shape
    assert errors.shape == expected_shape
    assert counts.shape == expected_shape


def test_save_sqw_writes_pixel_data(output_file: sqw.Sqw) -> None:
    metadata = output_file.read_data_block("data", "metadata")
    dnd = output_file.read_data_block("data", "nd_data")
    pix = output_file.read_data_block("pix", "data_wrap")

    assert pix.shape == (
        N_DETECTORS * N_PIXELS_PER_DETECTOR * N_ANGLES * ENERGY_BIN_SIZE,
        9,
    )

    n_pix = dnd[2].astype(int)
    check_pixels_in_bin_ranges(pix, metadata, n_pix)
    check_pixel_indices_in_ranges(pix)


def check_pixels_in_bin_ranges(
    pix: npt.NDArray[np.float32], metadata: sqw.SqwDndMetadata, n_pix: npt.NDArray[int]
) -> None:
    """Check that all pixels are within the bin edges defined in the dnd metadata."""
    img_range = metadata.axes.img_range
    n_bins = metadata.axes.n_bins_all_dims.values
    u_edges = [
        sc.linspace(f'u{i}', img_range[i][0], img_range[i][1], n_bins[i] + 1)
        for i in range(len(n_bins))
    ]

    bin_offsets = np.r_[0, np.cumsum(n_pix.flat)]
    for bin_index, (l, k, j, i) in enumerate(  # noqa: E741
        itertools.product(*(range(nb) for nb in n_bins[::-1]))
    ):
        pix_slice = slice(bin_offsets[bin_index], bin_offsets[bin_index + 1])
        if pix_slice.stop <= pix_slice.start:
            continue  # empty bin => no pixels to check
        for col, index in enumerate((i, j, k, l)):
            p = pix[pix_slice, col]
            p_min = p.min()  # actual range
            p_max = p.max()
            e_min = u_edges[col][index].value  # expected range
            e_max = u_edges[col][index + 1].value
            assert p_min >= e_min - 1e-6  # small offset to account for f32 rounding
            assert p_max < e_max + 1e-6


def check_pixel_indices_in_ranges(pix: npt.NDArray[np.float32]) -> None:
    """Check that all indices in the pixel data are in the correct ranges."""

    irun = pix[:, 4]
    idet = pix[:, 5]
    ien = pix[:, 6]

    # 1-based indices!
    assert irun.min() == 1
    assert irun.max() == N_ANGLES
    assert ien.min() == 1
    assert ien.max() == ENERGY_BIN_SIZE

    # We cannot check the values of idet directly because they are a non-contiguous
    # subset of all detector numbers. So just check that we have the correct number of
    # distinct detectors:
    assert np.unique(idet).size == N_DETECTORS * N_PIXELS_PER_DETECTOR
