# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import io
import re
from typing import Any

import pytest
import scipp as sc

import ess.dream.io.cif
from ess.dream.io.cif import CIFAuthor, CIFAuthors
from ess.powder.types import IofDspacing

# TODO many more


@pytest.fixture()
def iofd() -> IofDspacing:
    return IofDspacing(
        sc.DataArray(
            sc.array(dims=['dspacing'], values=[2.1, 3.2], variances=[0.3, 0.4]),
            coords={'dspacing': sc.linspace('dspacing', 0.1, 1.2, 3, unit='angstrom')},
        )
    )


def save_reduced_dspacing_to_str(*args: Any, **kwargs: Any) -> str:
    buffer = io.StringIO()
    ess.dream.io.cif.save_reduced_dspacing(*args, filename=buffer, **kwargs)
    buffer.seek(0)
    return buffer.read()


def test_save_reduced_dspacing_writes_contact_author(iofd: IofDspacing) -> None:
    authors = CIFAuthors(
        [
            CIFAuthor(
                name='Jane Doe',
                email='jane.doe@ess.eu',
                address='Partikelgatan, Lund',
                id_orcid='https://orcid.org/0000-0000-0000-0001',
            )
        ]
    )
    result = save_reduced_dspacing_to_str(iofd, authors=authors)
    assert "_audit_contact_author.name 'Jane Doe'" in result
    assert '_audit_contact_author.email jane.doe@ess.eu' in result
    assert "_audit_contact_author.address 'Partikelgatan, Lund'" in result
    assert (
        '_audit_contact_author.id_orcid https://orcid.org/0000-0000-0000-0001' in result
    )


def test_save_reduced_dspacing_writes_regular_author(iofd: IofDspacing) -> None:
    authors = CIFAuthors(
        [
            CIFAuthor(
                name='Jane Doe',
                email='jane.doe@ess.eu',
                address='Partikelgatan, Lund',
                id_orcid='https://orcid.org/0000-0000-0000-0001',
                is_contact=False,
            )
        ]
    )
    result = save_reduced_dspacing_to_str(iofd, authors=authors)
    assert "_audit_author.name 'Jane Doe'" in result
    assert '_audit_author.email jane.doe@ess.eu' in result
    assert "_audit_author.address 'Partikelgatan, Lund'" in result
    assert '_audit_author.id_orcid https://orcid.org/0000-0000-0000-0001' in result


def test_save_reduced_dspacing_writes_multiple_regular_authors(
    iofd: IofDspacing,
) -> None:
    authors = CIFAuthors(
        [
            CIFAuthor(
                name='Jane Doe',
                email='jane.doe@ess.eu',
                address='Partikelgatan, Lund',
                id_orcid='https://orcid.org/0000-0000-0000-0001',
                is_contact=False,
            ),
            CIFAuthor(
                name='Max Mustermann',
                email='mm@scipp.eu',
                id_orcid='https://orcid.org/0000-0000-0000-0002',
                is_contact=False,
            ),
        ]
    )
    result = save_reduced_dspacing_to_str(iofd, authors=authors)
    # The missing address for Max is currently broken because of
    # https://github.com/scipp/scippneutron/issues/547
    expected = """loop_
_audit_author.name
_audit_author.email
_audit_author.address
_audit_author.id_orcid
'Jane Doe' jane.doe@ess.eu 'Partikelgatan, Lund' https://orcid.org/0000-0000-0000-0001
'Max Mustermann' mm@scipp.eu  https://orcid.org/0000-0000-0000-0002
"""
    assert expected in result


def test_save_reduced_dspacing_writes_regular_author_role(iofd: IofDspacing) -> None:
    authors = CIFAuthors(
        [
            CIFAuthor(
                name='Jane Doe',
                role='measurement',
                is_contact=False,
            )
        ]
    )
    result = save_reduced_dspacing_to_str(iofd, authors=authors)

    author_pattern = r"""_audit_author.name 'Jane Doe'
_audit_author.id ([0-9a-f]+)"""
    author_match = re.search(author_pattern, result)
    assert author_match is not None
    author_id = author_match.group(1)

    expected = rf"""loop_
_audit_author_role.id
_audit_author_role.role
{author_id} measurement"""

    assert expected in result
