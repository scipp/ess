# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import platform
from datetime import datetime

from orsopy import fileio

from .. import __version__


def make_orso(
    owner: fileio.base.Person,
    sample: fileio.data_source.Sample,
    creator: fileio.base.Person,
    reduction_script: str,
) -> fileio.orso.Orso:
    """
    Generate the base Orso object for the Amor instrument.
    Populate the Orso object for metadata storage.

    Parameters
    ----------
    owner:
        The owner of the data set, i.e. the main proposer of the measurement.
    sample:
        A description of the sample.
    creator:
        The creator of the reduced data, the person responsible for the
        reduction process.
    reduction_script:
        The script or notebook used for reduction.

    Returns
    -------
    :
        Orso object with the default parameters for the Amor instrument.
    """
    orso = fileio.orso.Orso.empty()
    orso.data_source.experiment.probe = 'neutrons'
    orso.data_source.experiment.facility = 'Paul Scherrer Institut'
    orso.data_source.measurement.scheme = 'angle- and energy-dispersive'
    orso.reduction.software = fileio.reduction.Software(
        'scipp-ess', __version__, platform.platform()
    )
    orso.reduction.timestep = datetime.now()
    orso.reduction.corrections = []
    orso.reduction.computer = platform.node()
    orso.columns = [
        fileio.base.Column('Qz', '1/angstrom', 'wavevector transfer'),
        fileio.base.Column('R', None, 'reflectivity'),
        fileio.base.Column('sR', None, 'standard deivation of reflectivity'),
        fileio.base.Column(
            'sQz', '1/angstrom', 'standard deviation of wavevector transfer resolution'
        ),
    ]
    orso.data_source.owner = owner
    orso.data_source.sample = sample
    orso.reduction.creator = creator
    orso.reduction.script = reduction_script
    return orso
