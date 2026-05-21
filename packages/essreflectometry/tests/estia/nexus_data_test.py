# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import sciline
import scipp as sc
import scipp.testing
from ess.reduce.normalization import (
    normalize_by_monitor_histogram,
    normalize_by_monitor_integrated,
)
from ess.reduce.uncertainty import UncertaintyBroadcastMode
from orsopy import fileio

from ess.estia import EstiaWorkflow
from ess.estia.corrections import RunNormalization
from ess.estia.data import (
    estia_mcstas_nexus_reference_example,
    estia_mcstas_nexus_sample_example,
    estia_wavelength_lookup_table,
)
from ess.estia.types import WavelengthMonitor
from ess.reflectometry import orso, supermirror
from ess.reflectometry.types import (
    BeamDivergenceLimits,
    Filename,
    LookupTableFilename,
    ProtonCharge,
    QBins,
    ReducibleData,
    ReferenceRun,
    ReflectivityOverQ,
    SampleRun,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)


def _make_estia_pipeline(
    *, run_norm: RunNormalization = RunNormalization.none
) -> sciline.Pipeline:
    wf = EstiaWorkflow(run_norm=run_norm)
    wf[Filename[ReferenceRun]] = estia_mcstas_nexus_reference_example()

    wf[YIndexLimits] = sc.scalar(35), sc.scalar(64)
    wf[ZIndexLimits] = sc.scalar(0), sc.scalar(48 * 32)
    wf[BeamDivergenceLimits] = sc.scalar(-1.0, unit='deg'), sc.scalar(1.0, unit='deg')
    wf[WavelengthBins] = sc.geomspace('wavelength', 3.5, 12, 2001, unit='angstrom')
    wf[QBins] = sc.geomspace('Q', 0.005, 0.1, 200, unit='1/angstrom')

    wf[supermirror.CriticalEdge] = sc.scalar(float('inf'), unit='1/angstrom')
    wf[supermirror.Alpha] = sc.scalar(0.25 / 0.088, unit=sc.units.angstrom)
    wf[supermirror.MValue] = sc.scalar(5, unit=sc.units.dimensionless)

    wf[LookupTableFilename] = estia_wavelength_lookup_table()

    wf[ProtonCharge[SampleRun]] = sc.DataArray(
        sc.array(dims=('time',), values=[]),
        coords={'time': sc.array(dims=('time',), values=[], unit='s')},
    )
    wf[ProtonCharge[ReferenceRun]] = sc.DataArray(
        sc.array(dims=('time',), values=[]),
        coords={'time': sc.array(dims=('time',), values=[], unit='s')},
    )
    wf[orso.OrsoCreator] = orso.OrsoCreator(
        fileio.base.Person(
            name="Max Mustermann",
            affiliation="European Spallation Source ERIC",
            contact="max.mustermann@ess.eu",
        )
    )
    wf[orso.OrsoExperiment] = orso.OrsoExperiment(
        fileio.data_source.Experiment(
            title='McStas run',
            instrument='Estia',
            facility='ESS',
            start_date=datetime(2025, 3, 20, tzinfo=ZoneInfo("Europe/Stockholm")),
            probe='neutron',
        )
    )
    wf[orso.OrsoOwner] = orso.OrsoOwner(
        fileio.base.Person(
            name='John Doe',
            contact='john.doe@ess.eu',
            affiliation='ESS',
        )
    )
    wf[orso.OrsoSample] = orso.OrsoSample(fileio.data_source.Sample.empty())
    return wf


@pytest.fixture
def estia_pipeline() -> sciline.Pipeline:
    return _make_estia_pipeline()


def test_compute_reducible_data(estia_pipeline: sciline.Pipeline):
    estia_pipeline[Filename[SampleRun]] = estia_mcstas_nexus_sample_example(
        'Ni/Ti-multilayer'
    )[0]
    da = estia_pipeline.compute(ReducibleData[SampleRun])
    assert da.dims == ('strip', 'blade', 'wire')
    assert da.shape == (64, 48, 32)
    assert 'position' in da.coords
    assert 'sample_rotation' in da.coords
    assert 'detector_rotation' in da.coords
    assert 'theta' in da.coords
    assert 'wavelength' in da.bins.coords
    assert 'Q' in da.bins.coords


@pytest.mark.parametrize(
    ("run_norm", "normalize"),
    [
        (RunNormalization.monitor_histogram, normalize_by_monitor_histogram),
        (RunNormalization.monitor_integrated, normalize_by_monitor_integrated),
    ],
)
def test_compute_reducible_data_with_monitor(
    estia_pipeline: sciline.Pipeline,
    run_norm: RunNormalization,
    normalize,
):
    filename = estia_mcstas_nexus_sample_example('Ni/Ti-multilayer')[0]
    estia_pipeline[Filename[SampleRun]] = filename
    without_monitor = estia_pipeline.compute(ReducibleData[SampleRun])

    wf = _make_estia_pipeline(run_norm=run_norm)
    wf[Filename[SampleRun]] = filename
    wf[WavelengthMonitor[SampleRun]] = sc.DataArray(
        sc.array(dims=['wavelength'], values=[30.0], variances=[1.0]),
        coords={'wavelength': sc.linspace('wavelength', 0, 15, 2, unit='angstrom')},
    )
    with_monitor = wf.compute(ReducibleData[SampleRun])
    scipp.testing.assert_allclose(
        normalize(
            without_monitor,
            monitor=wf.compute(WavelengthMonitor[SampleRun]),
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.drop,
        ),
        with_monitor,
    )


def test_can_compute_reflectivity_curve(estia_pipeline: sciline.Pipeline):
    estia_pipeline[Filename[SampleRun]] = estia_mcstas_nexus_sample_example(
        'Ni/Ti-multilayer'
    )[0]
    r = estia_pipeline.compute(ReflectivityOverQ)
    assert "Q" in r.coords
    assert "Q_resolution" in r.coords


def test_orso_pipeline(estia_pipeline: sciline.Pipeline):
    estia_pipeline[Filename[SampleRun]] = estia_mcstas_nexus_sample_example(
        'Ni/Ti-multilayer'
    )[0]
    res = estia_pipeline.compute(orso.OrsoIofQDataset)
    assert res.info.data_source.experiment.instrument == "Estia"
    assert res.info.reduction.software.name == "ess.reflectometry"
    assert res.info.reduction.corrections == [
        "chopper ToF correction",
        "footprint correction",
        "supermirror calibration",
    ]
    assert res.data.ndim == 2
    assert res.data.shape[1] == 4
    assert np.all(res.data[:, 1] >= 0)
    assert np.isfinite(res.data).all()


def test_save_reduced_orso_file(estia_pipeline: sciline.Pipeline, output_folder: Path):
    estia_pipeline[Filename[SampleRun]] = estia_mcstas_nexus_sample_example(
        'Ni/Ti-multilayer'
    )[0]
    res = estia_pipeline.compute(orso.OrsoIofQDataset)
    fileio.orso.save_orso(
        datasets=[res], fname=output_folder / 'estia_reduced_iofq.ort'
    )
