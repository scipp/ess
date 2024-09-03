# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""CIF writer for DREAM."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

import scipp as sc
from scippneutron.io import cif

from ess.powder.types import IofDspacing, OutFilename


@dataclass(kw_only=True)
class CIFAuthor:
    name: str
    address: str | None = None
    email: str | None = None
    id_orcid: str | None = None
    is_contact: bool = True
    role: str | None = None


@dataclass()
class CIFAuthors:
    authors: Sequence[CIFAuthor] = field(default_factory=list)


def save_reduced_dspacing(
    da: IofDspacing, *, filename: OutFilename, authors: CIFAuthors
) -> IofDspacing:
    to_save = da if da.bins is None else da.hist()
    cif.save_cif(filename, _make_dspacing_block(to_save, authors=authors))
    return da


def _make_dspacing_block(da: sc.DataArray, authors: CIFAuthors) -> cif.Block:
    block = cif.Block(
        'reduced_dspacing',
        [
            _make_audit_chunk(),
            _make_source_chunk(),
            *_make_author_chunks(authors),
            _make_dspacing_loop(da),
        ],
    )
    return block


def _make_dspacing_loop(da: sc.DataArray) -> cif.Loop:
    return cif.Loop(
        {
            'pd_proc.d_spacing': sc.midpoints(da.coords['dspacing']),
            'pd_proc.intensity_net': sc.values(da.data),
            'pd_proc.intensity_su': sc.stddevs(da.data),
        },
        schema=cif.PD_SCHEMA,
    )


def _make_source_chunk() -> cif.Chunk:
    return cif.Chunk(
        {
            # TODO diffrn_source.current once we have it
            # TODO diffrn_source.power: can we deduce it?
            'diffrn_radiation.probe': 'neutron',
            'diffrn_source.beamline': 'DREAM',
            'diffrn_source.device': 'spallation',
            'diffrn_source.facility': 'ESS',
        },
        schema=cif.CORE_SCHEMA,
    )


def _make_audit_chunk() -> cif.Chunk:
    from ess.dream import __version__

    return cif.Chunk(
        {
            'audit.creation_date': datetime.now(timezone.utc).isoformat(
                timespec='seconds'
            ),
            'audit.creation_method': f'Written by ess.dream v{__version__}',
        },
        schema=cif.CORE_SCHEMA,
    )


def _make_author_chunks(authors: CIFAuthors) -> list[cif.Chunk | cif.Loop]:
    contact = [author for author in authors.authors if author.is_contact]
    regular = [author for author in authors.authors if not author.is_contact]

    results = []
    roles = {}
    for aut, cat in zip(
        (contact, regular), ('audit_contact_author', 'audit_author'), strict=True
    ):
        if not aut:
            continue
        data, rols = _serialize_authors(aut, cat)
        results.append(data)
        roles.update(rols)
    if roles:
        results.append(_serialize_roles(roles))

    return results


def _serialize_authors(
    authors: list[CIFAuthor], category: str
) -> tuple[cif.Chunk | cif.Loop, dict[str, str]]:
    fields = {
        f'{category}.{key}': f
        for key in ('name', 'email', 'address', 'id_orcid')
        if any(f := [getattr(a, key) or '' for a in authors])
    }

    roles = {uuid.uuid4().hex: a.role for a in authors}
    if any(roles.values()):
        fields[f'{category}.id'] = list(roles.keys())
    roles = {key: val for key, val in roles.items() if val}

    if len(authors) == 1:
        return cif.Chunk(
            {key: val[0] for key, val in fields.items()},
            schema=cif.CORE_SCHEMA,
        ), roles
    return cif.Loop(
        {key: sc.array(dims=['author'], values=val) for key, val in fields.items()},
        schema=cif.CORE_SCHEMA,
    ), roles


def _serialize_roles(roles: dict[str, str]) -> cif.Loop:
    return cif.Loop(
        {
            'audit_author_role.id': sc.array(dims=['role'], values=list(roles)),
            'audit_author_role.role': sc.array(
                dims=['role'], values=list(roles.values())
            ),
        },
        schema=cif.CORE_SCHEMA,
    )
