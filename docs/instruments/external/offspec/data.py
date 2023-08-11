import scitacean
from scitacean import Dataset
from datetime import datetime


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
        creation_location = "ISIS Neutron and Muon Source",
        instrument_id = "OFFSPEC"
    )
    file = './direct_beam.nxs'
    direct_beam_dataset.add_local_files(file, base_path='.')
    return client.upload_new_dataset_now(direct_beam_dataset)


def get_sample(client: scitacean.Client, direct_beam_uploaded: Dataset):
    sample_dataset = Dataset(
        name="OFFSPEC Sample Data",
        description="Raw OFFSPEC data from quartz and copper at the air-silicon interface.",
        type="raw",

        owner_group="ess",
        access_groups=["offspec"],

        owner="Joshanial F. K. Cooper",
        principal_investigator="Joshanial F. K. Cooper",
        contact_email="jos.cooper@stfc.ac.uk",
        end_time=datetime.now(),

        data_format="ISIS NeXus file",
        creation_location = "ISIS Neutron and Muon Source",
        instrument_id = "OFFSPEC"
    )
    file = './sample.nxs'
    sample_dataset.add_local_files(file, base_path='.')
    sample_dataset.meta['direct_beam_pid'] = str(direct_beam_uploaded.pid)
    sample_dataset.meta['sample_name'] = "QCS sample"
    sample_dataset.meta['sample_category'] = "gas/solid"
    sample_dataset.meta['sample_composition'] = "Air | Si(790 A) | Cu(300 A) | SiO2"
    return client.upload_new_dataset_now(sample_dataset)