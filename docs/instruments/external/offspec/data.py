import scitacean
from pathlib import Path
from scitacean import Dataset
from datetime import datetime

# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
_version = '1'


def _make_pooch():
    import pooch
    return pooch.create(
        path=pooch.os_cache('ess/offspec'),
        env='ESS_OFFSPEC_DATA_DIR',
        base_url='https://public.esss.dk/groups/scipp/ess/offspec/{version}/',
        version=_version,
        registry={
            "direct_beam.nxs": "md5:e929d3419b13c3ffa4a5545ec54f9044",
            "sample.nxs": "md5:f18a8122706201df8150e7556ae6eb59",
            "reduced_mantid.xye": "md5:1f372f51d2cefb8dee302cf0093b684f"
        })


_pooch = _make_pooch()


def get_path(name: str) -> str:
    """
    Return the path to a data file bundled with scippneutron.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    return _pooch.fetch(name)


def get_direct_beam(client: scitacean.Client):
    direct_beam_dataset = Dataset(
        name="OFFSPEC Direct Beam Data",
        description="Raw OFFSPEC data from the direct beam.",
        type="raw",

        owner_group="ess",
        access_groups=["offspec"],

        owner="Joshanial F. K. Cooper",
        principal_investigator="Joshanial F. K. Cooper",
        contact_email="jos.cooper@stfc.ac.uk",
        end_time=datetime.now(),

        data_format="ISIS NeXus file",
        creation_location="ISIS Neutron and Muon Source",
        instrument_id="OFFSPEC"
    )
    file = Path(get_path('direct_beam.nxs'))
    direct_beam_dataset.add_local_files(str(file), base_path=str(file.parents[0]))
    return client.upload_new_dataset_now(direct_beam_dataset)


def get_sample(client: scitacean.Client, direct_beam_uploaded: Dataset):
    sample_dataset = Dataset(
        name="OFFSPEC Sample Data",
        description="Raw OFFSPEC data from quartz and "
                    "copper at the air-silicon interface.",
        type="raw",

        owner_group="ess",
        access_groups=["offspec"],

        owner="Joshanial F. K. Cooper",
        principal_investigator="Joshanial F. K. Cooper",
        contact_email="jos.cooper@stfc.ac.uk",
        end_time=datetime.now(),

        data_format="ISIS NeXus file",
        creation_location="ISIS Neutron and Muon Source",
        instrument_id="OFFSPEC"
    )
    file = Path(get_path('sample.nxs'))
    sample_dataset.add_local_files(str(file), base_path=str(file.parents[0]))
    sample_dataset.meta['direct_beam_pid'] = str(direct_beam_uploaded.pid)
    sample_dataset.meta['sample_name'] = "QCS sample"
    sample_dataset.meta['sample_category'] = "gas/solid"
    sample_dataset.meta['sample_composition'] = "Air | Si(790 A) | Cu(300 A) | SiO2"
    return client.upload_new_dataset_now(sample_dataset)


def get_mantid():
    return Path(get_path('reduced_mantid.xye'))
