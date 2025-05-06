# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import re

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
            "amor2024n001632.hdf": "md5:2253f0ec6d2e96a986a6aa35d43a7480",
            "amor2024n001634.hdf": "md5:7cdd87bbd96fb3fb1e046800a9b1d77e",
            "amor2024n001635.hdf": "md5:fb9eb0e7b803c13f1804d085b3b0058f",
            "amor2024n001636.hdf": "md5:f7deb51d22652d1f5d0e4b51927af5a3",
            "amor2024n001637.hdf": "md5:06e5957d34d82035cfece40cbbf47d7a",
            "amor2024n001638.hdf": "md5:1193fe808af2afeb0f48d3b0022fe40b",
            "amor2024n001639.hdf": "md5:ff24e31a07c4020927aaa6df9f2ee05f",
            "amor2024n001640.hdf": "md5:051335fea1c369322a2328d530dedb77",
            "amor2024n001641.hdf": "md5:d543b5890b63707cf8d7f6666e8830a4",
            "amor2024n001642.hdf": "md5:b1474e32cc64371e1005f44f2c5b6ae7",
            "amor2024n004079.hdf": "md5:5bf1dabc2ff902a57ed7593903c8e1a5",
            "amor2024n004080.hdf": "md5:7dcaa5da00b7eedc178b9e55209bfbce",
            "amor2024n004081.hdf": "md5:2f4f46f0e56ab75aad0c2933060df504",
            "amor2024n004083.hdf": "md5:ffd13420e44cbf966b94bd532e293c1c",
            "amor2024n004084.hdf": "md5:5041a9486b3cd6407a9b9f44104c9b54",
            "amor2024n004085.hdf": "md5:152ac63bad6f3b2b5cf1c4e4c7df5e30",
            "amor2024n004152.hdf": "md5:244459be7a3caeac523c289815c5b7dc",
            "amor2024n004154.hdf": "md5:6184553f795fd2b230baab9a421da9e4",
        },
    )


_pooch = _make_pooch()


def amor_old_sample_run() -> str:
    return _pooch.fetch("sample.nxs")


def amor_old_reference_run() -> str:
    return _pooch.fetch("reference.nxs")


def amor_run(number: int | str) -> str:
    fnames = [
        name
        for name in _pooch.registry.keys()
        if re.match(f'amor\\d{{4}}n{int(number):06d}.hdf', name)
    ]
    if len(fnames) != 1:
        raise ValueError(f'Expected exactly one matching file, found {len(fnames)}')
    return _pooch.fetch(fnames[0])


def amor_psi_software_result(number: int | str) -> str:
    return _pooch.fetch(f"{int(number):03d}.Rqz.ort")


__all__ = [
    "amor_psi_software_result",
    "amor_run",
]
