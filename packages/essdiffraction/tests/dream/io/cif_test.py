# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import datetime
import io

import ess.dream.io.cif
import numpy as np
import pytest
import scipp as sc
import scipp.testing
from ess.powder.calibration import OutputCalibrationData
from ess.powder.types import (
    Beamline,
    CIFAuthors,
    IntensityTof,
    Measurement,
    ReducerSoftware,
    Software,
)
from scippneutron.io import cif
from scippneutron.metadata import ESS_SOURCE, Person


@pytest.fixture
def ioftof() -> IntensityTof:
    return IntensityTof(
        sc.DataArray(
            sc.array(dims=['tof'], values=[2.1, 3.2, 1.6], variances=[0.3, 0.4, 0.1]),
            coords={
                'tof': sc.array(dims=['tof'], values=[0.1, 0.3, 0.5, 0.7], unit='us')
            },
            masks={'bad': sc.array(dims=['tof'], values=[False, True, False])},
        )
    )


@pytest.fixture
def cal() -> OutputCalibrationData:
    return OutputCalibrationData(
        {
            0: sc.scalar(0.2, unit='us'),
            1: sc.scalar(1.2, unit='us/angstrom'),
            2: sc.scalar(-1.4, unit='us/angstrom^2'),
        }
    )


def save_reduced_tof_to_str(cif_: cif.CIF) -> str:
    buffer = io.StringIO()
    cif_.save(buffer)
    buffer.seek(0)
    return buffer.read()


def test_save_reduced_tof(ioftof: IntensityTof, cal: OutputCalibrationData) -> None:
    from ess.dream import __version__

    author = Person(name='John Doe', corresponding=True)
    cif_ = ess.dream.io.cif.prepare_reduced_tof_cif(
        ioftof,
        authors=CIFAuthors([author]),
        beamline=Beamline(
            name="DREAM",
            facility="ESS",
            site="ESS",
        ),
        source=ESS_SOURCE,
        measurement=Measurement(
            title="Test measurement",
            start_time=datetime.datetime(2026, 1, 2, 14, 58, 2, tzinfo=datetime.UTC),
        ),
        reducers=ReducerSoftware(
            [
                Software.from_package_metadata('ess.diffraction'),
                Software.from_package_metadata('ess.dream'),
                Software.from_package_metadata('ess.powder'),
                Software.from_package_metadata('scippneutron'),
                Software.from_package_metadata('scipp'),
            ]
        ),
        calibration=cal,
    )
    result = save_reduced_tof_to_str(cif_)

    assert "_audit_contact_author.name 'John Doe'" in result
    assert f"_computing.diffrn_reduction\n'ess.diffraction {__version__}'" in result
    assert f"ess.dream {__version__}" in result
    assert f"ess.powder {__version__}" in result
    assert '_diffrn_source.beamline DREAM' in result
    assert "_sc_meas.title 'Test measurement'" in result
    assert "_pd_meas.datetime_initiated 2026-01-02T14:58:02" in result
    assert 'ZERO 0 0.2' in result
    assert 'DIFC 1 1.2' in result
    assert 'DIFA 2 -1.4' in result

    loop_header = """loop_
_pd_data.point_id
_pd_meas.time_of_flight
_pd_proc.intensity_norm
_pd_proc.intensity_norm_su
"""
    assert loop_header in result


def test_save_reduced_tof_writes_excpected_data(
    ioftof: IntensityTof, cal: OutputCalibrationData
) -> None:
    cif_ = ess.dream.io.cif.prepare_reduced_tof_cif(
        ioftof,
        authors=CIFAuthors([]),
        beamline=Beamline(
            name="DREAM",
        ),
        source=ESS_SOURCE,
        measurement=Measurement(
            title="Test measurement",
        ),
        reducers=ReducerSoftware([]),
        calibration=cal,
    )
    result = save_reduced_tof_to_str(cif_)

    loop_header = """loop_
_pd_data.point_id
_pd_meas.time_of_flight
_pd_proc.intensity_norm
_pd_proc.intensity_norm_su
"""
    data_table = result[result.index(loop_header) + len(loop_header) :]
    _, tof, val, std = np.loadtxt(io.StringIO(data_table), delimiter=' ').T
    loaded = sc.DataArray(
        sc.array(dims=['tof'], values=val, variances=std**2),
        coords={'tof': sc.array(dims=['tof'], values=tof, unit='us')},
    )

    expected = sc.DataArray(
        sc.array(dims=['tof'], values=[2.1, 0.0, 1.6], variances=[0.3, 0.0, 0.1]),
        coords={'tof': sc.array(dims=['tof'], values=[0.2, 0.4, 0.6], unit='us')},
    )
    # Can't be identical because of conversion to standard deviations
    sc.testing.assert_allclose(loaded, expected)
