# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

# Writing an SQW file is fairly slow.
# So the tests in this module use module-scoped fixtures to write a single file
# that is shared between all tests.
# Function-scoped fixtures allow accessing that file for reading.

from collections.abc import Generator
from pathlib import Path

import numpy as np
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
    TimeOfFlightLookupTable,
)

N_DETECTORS = 3
N_ANGLES = 180

BIN_SIZES = {'Qx': 6, 'Qy': 7, 'Qz': 8, 'energy_transfer': 9}
ENERGY_BIN_SIZE = 13

# Q projections
U = sc.vector([1, 0, 0], unit="1/angstrom")
V = sc.vector([0, 1, 0], unit="1/angstrom")
W = sc.vector([0, 0, 1], unit="1/angstrom")


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
    wf[TimeOfFlightLookupTable] = sc.io.load_hdf5(tof_lookup_table_simulation())
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
