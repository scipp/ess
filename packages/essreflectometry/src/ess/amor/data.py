# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from ..reflectometry.types import Filename, ReferenceRun, SampleRun

_version = "2"


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache("ess/amor"),
        env="ESS_AMOR_DATA_DIR",
        base_url="https://public.esss.dk/groups/scipp/ess/amor/{version}/",
        version=_version,
        registry={
            "reference.nxs": "md5:56d493c8051e1c5c86fb7a95f8ec643b",
            "sample.nxs": "md5:4e07ccc87b5c6549e190bc372c298e83",
            # Amor NeXus files
            # 608 to 611 are measurements on the same sample at different
            # sample rotations.
            # 612 and 613 are unknown to me.
            # 614 is a reference measurement on a super mirror.
            # The sample rotation values written in the files are wrong,
            # the real values are the following (all expressed in deg):
            #   608: 0.85,
            #   609: 2.25,
            #   610: 3.65,
            #   611: 5.05,
            #   612: 0.65,
            #   613: 0.65,
            #   614: 0.65,
            # The chopper phase offset value written in the files is wrong
            # the real value is -7.5 deg
            "amor2023n000608.hdf": "md5:e3a8b5b8495bb9ab2173848096df49d6",
            "amor2023n000609.hdf": "md5:d899d65f684cade2905f203f7d0fb326",
            "amor2023n000610.hdf": "md5:c9367d49079edcd17fa0b98e33326b05",
            "amor2023n000611.hdf": "md5:9da41177269faac0d936806393427837",
            "amor2023n000612.hdf": "md5:602f1bfcdbc1f618133c93a117d05f12",
            "amor2023n000613.hdf": "md5:ba0fbcbf0b45756a269eb3e943371ced",
            "amor2023n000614.hdf": "md5:18e8a755d6fd671758fe726de058e707",
            # Reflectivity curves obtained by applying Jochens Amor
            # software @ https://github.com/jochenstahn/amor.git
            # (repo commit hash 05e35ca4e05436d7c69ff6e19f32bc1915cbb5d0).
            # to the above files.
            "608.Rqz.ort": "md5:e7e7d63a1ac1e727e9b2f12dc78a77ce",
            "609.Rqz.ort": "md5:3cb3fd11a743594f52a10f71b122b71a",
            "610.Rqz.ort": "md5:66d43993e76801655a1d629cb976abde",
            "611.Rqz.ort": "md5:0c51e8ac5c00041434417673be186151",
            "612.Rqz.ort": "md5:d785d27151e7f1edc05e86d35bef6a63",
            "613.Rqz.ort": "md5:e999c85f7a47665c4ddd1538b19d402d",
        },
    )


_pooch = _make_pooch()


def amor_old_sample_run() -> Filename[SampleRun]:
    return Filename[SampleRun](_pooch.fetch("sample.nxs"))


def amor_old_reference_run() -> Filename[ReferenceRun]:
    return Filename[ReferenceRun](_pooch.fetch("reference.nxs"))


def amor_reference_run() -> Filename[ReferenceRun]:
    return Filename[ReferenceRun](_pooch.fetch("amor2023n000614.hdf"))


def amor_sample_run(number: int | str) -> Filename[SampleRun]:
    return Filename[SampleRun](_pooch.fetch(f"amor2023n{int(number):06d}.hdf"))


def amor_psi_software_result(number: int | str) -> Filename[SampleRun]:
    return Filename[SampleRun](_pooch.fetch(f"{int(number):03d}.Rqz.ort"))


__all__ = [
    "amor_psi_software_result",
    "amor_reference_run",
    "amor_sample_run",
]
