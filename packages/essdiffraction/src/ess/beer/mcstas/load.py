# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Loaders and helpers for BEER McStas simulation data."""

from pathlib import Path

import mcstastox
import scipp as sc
import scippnexus as snx
from ess.powder.types import CaveMonitor, RunType, WavelengthMonitor

from ess.reduce.nexus.types import DiskChoppers, Position
from ess.reduce.unwrap.types import DetectorLtotal

from ..types import DetectorBank, Filename, RawDetector
from .beamline import ModulationMode, PulseShapingMode, simulation_choppers

# The Beer McStas files use the convention that t=0 corresponds
# to 1.6ms after the start of the pulse.
# But in the real files the convention will be that event_time_offset=0
# corresponds to the start of the pulse.
_MCSTAS_T_OFFSET = sc.scalar(1.6, unit='ms')

__all__ = [
    'chopper_mode_from_mcstas_mode',
    'load_beer_mcstas',
    'load_beer_mcstas_monitor',
    'load_beer_mcstas_monitor_provider',
    'load_beer_mcstas_provider',
    'mcstas_choppers',
    'mcstas_detector_ltotal',
    'mcstas_providers',
    'mcstas_sample_position',
    'mcstas_source_position',
]


def _detector_components(data: mcstastox.Read, bank: DetectorBank) -> list[str]:
    components = data.get_components_with_ids()

    prefix = f'nd_mantid_{bank.name}_'
    panels = [name for name in components if name.lower().startswith(prefix)]
    if panels:
        return panels

    number = 1 if bank is DetectorBank.south else 2
    short_name = 'S' if bank is DetectorBank.south else 'N'
    candidates = (
        f'nD_Mantid_{short_name}2_2D',
        f'nD_Mantid_{number}',
        f'nD_Mantid{number}',
    )
    components_by_name = {name.lower(): name for name in components}
    for candidate in candidates:
        if component := components_by_name.get(candidate.lower()):
            return [component]

    available = ', '.join(sorted(components))
    raise ValueError(
        f'Could not find the {bank.value} detector bank in the McStas file. '
        f'Components with pixel IDs: {available}'
    )


def _load_events(data: mcstastox.Read, components: list[str]) -> sc.DataArray:
    """Load weighted events and pixel positions for selected components only.

    Loading only the selected components is important for older BEER simulations:
    they contain auxiliary detectors with incomplete geometry which cannot be exported
    together with the Mantid detector banks.
    """
    event_data = data.get_event_data(
        variables=['p', 't', 'id'], component_name=components, filter_zeros=True
    )
    weights = event_data['p']
    events = sc.DataArray(
        sc.array(
            dims=['events'],
            values=weights,
            variances=weights**2,
            unit='counts',
        ),
        coords={
            'pixel_id': sc.array(
                dims=['events'], values=event_data['id'].astype('int32')
            ),
            't': sc.array(dims=['events'], values=event_data['t'], unit='s'),
        },
    ).group('pixel_id')

    position_tables = []
    for component in components:
        positions = data.get_component_global(component)
        begin, end = data.pixel_range[component]
        position_tables.append(
            sc.DataArray(
                sc.vectors(dims=['pixel_id'], values=positions, unit='m'),
                coords={
                    'pixel_id': sc.arange(
                        'pixel_id', int(begin), int(end) + 1, dtype='int32'
                    )
                },
            )
        )

    position_table = sc.sort(sc.concat(position_tables, dim='pixel_id'), key='pixel_id')
    events.coords['position'] = sc.lookup(
        position_table, dim='pixel_id', mode='previous'
    )[events.coords['pixel_id']]
    return events


def load_beer_mcstas(
    filename: str | Path,
    bank: DetectorBank,
) -> sc.DataArray:
    """Load a detector bank from a BEER McStas file."""
    if not isinstance(bank, DetectorBank):
        raise ValueError('bank must be either DetectorBank.north or DetectorBank.south')

    filename = Path(filename)
    with mcstastox.Read(filename.parent, filename.name) as data:
        mode = data.file['entry1/simulation/Param/mode'][0]
        mode = mode.decode() if isinstance(mode, bytes) else str(mode)
        events = _load_events(data, _detector_components(data, bank))

    events.coords['mode'] = sc.scalar(mode)
    events.bins.coords['event_time_offset'] = (
        events.bins.coords.pop('t') + _MCSTAS_T_OFFSET.to(unit='s')
    ) % sc.scalar(1 / 14, unit='s')
    return events


def _to_edges(centers: sc.Variable) -> sc.Variable:
    interior_edges = sc.midpoints(centers)
    return sc.concat(
        [
            2 * centers[0] - interior_edges[0],
            interior_edges,
            2 * centers[-1] - interior_edges[-1],
        ],
        dim=centers.dim,
    )


def load_beer_mcstas_monitor(filename: str | Path) -> sc.DataArray:
    """Load the BEER McStas wavelength monitor."""
    filename = Path(filename)
    with mcstastox.Read(filename.parent, filename.name) as data:
        component = next(
            name
            for name in ('Lmon_hereon', 'Lmon')
            if name in data.get_components_with_data()
        )
        histogram = data.file_object.get_info_entry(component)
        wavelengths = next(
            values
            for name, values in histogram.items()
            if name not in {'data', 'errors', 'ncount'}
        )
        da = sc.DataArray(
            sc.array(
                dims=['wavelength'],
                values=histogram['data'][:],
                variances=histogram['errors'][:],
                unit='counts',
            ),
            coords={
                'wavelength': _to_edges(
                    sc.array(
                        dims=['wavelength'], values=wavelengths[:], unit='angstrom'
                    )
                ),
                'ncount': sc.array(
                    dims=['wavelength'], values=histogram['ncount'][:], unit='counts'
                ),
                'position': sc.vector(
                    data.get_global_component_coordinates(component), unit='m'
                ),
            },
        )
    return da


def load_beer_mcstas_provider(
    fname: Filename[RunType], bank: DetectorBank
) -> RawDetector[RunType]:
    """Sciline provider for loading BEER McStas detector data."""
    return load_beer_mcstas(fname, bank)


def load_beer_mcstas_monitor_provider(
    fname: Filename[RunType],
) -> WavelengthMonitor[RunType, CaveMonitor]:
    """Sciline provider for loading the BEER McStas wavelength monitor."""
    return load_beer_mcstas_monitor(fname)


_MCSTAS_CHOPPER_MODES = {
    '3': PulseShapingMode.ps0,
    '4': PulseShapingMode.ps1,
    '5': PulseShapingMode.ps2,
    '6': PulseShapingMode.ps3,
    '15': PulseShapingMode.ds1,
    '7': ModulationMode.m0,
    '8': ModulationMode.m1,
    '9': ModulationMode.m2,
    '10': ModulationMode.m3,
    '14': ModulationMode.ds0,
    '16': ModulationMode.m4,
}


def chopper_mode_from_mcstas_mode(mode: str) -> PulseShapingMode | ModulationMode:
    """Return the BEER chopper mode matching a McStas mode value."""
    mode = str(mode)
    if (chopper_mode := _MCSTAS_CHOPPER_MODES.get(mode)) is not None:
        return chopper_mode

    normalized = mode.upper()
    for chopper_mode in (*PulseShapingMode, *ModulationMode):
        if normalized == chopper_mode.value:
            return chopper_mode

    raise ValueError(f'Mode {mode} is not a known BEER chopper mode.')


def mcstas_source_position(
    fname: Filename[RunType], sample_position: Position[snx.NXsample, RunType]
) -> Position[snx.NXsource, RunType]:
    """Return the effective source position for the McStas geometry.

    ``sourceMantid`` is close to the choppers and its direction towards the sample
    is the incident-beam direction at the sample. It is not the moderator, however,
    so using it directly would make the source-to-sample distance too short.
    Conversely, the line from ``Origin`` to the sample has the correct length but
    not the incident direction because the guide is curved.

    The coordinate transformations use the source-to-sample vector both to define
    the incident beam and, through its length, to define ``Ltotal``. We therefore
    construct an effective source whose direction is given by ``sourceMantid`` and
    whose distance from the sample is given by ``Origin``.
    """
    fname = Path(fname)
    with mcstastox.Read(fname.parent, fname.name) as data:
        source_mantid = sc.vector(
            data.get_global_component_coordinates('sourceMantid'), unit='m'
        )
        moderator_position = sc.vector(
            data.get_global_component_coordinates('Origin'), unit='m'
        )
    incident_beam = sample_position - source_mantid
    return sample_position - incident_beam * (
        sc.norm(sample_position - moderator_position) / sc.norm(incident_beam)
    )


def mcstas_sample_position(
    fname: Filename[RunType],
) -> Position[snx.NXsample, RunType]:
    """Return the sample position from a BEER McStas file."""
    fname = Path(fname)
    with mcstastox.Read(fname.parent, fname.name) as data:
        return sc.vector(
            data.get_global_component_coordinates('sampleMantid'), unit='m'
        )


def mcstas_choppers(
    da: RawDetector[RunType],
    source_position: Position[snx.NXsource, RunType],
) -> DiskChoppers[RunType]:
    """Return BEER choppers for the McStas mode in the data."""
    mode = chopper_mode_from_mcstas_mode(da.coords['mode'].value)
    return simulation_choppers(mode, source_position)


def mcstas_detector_ltotal(
    da: RawDetector[RunType],
    source_position: Position[snx.NXsource, RunType],
    sample_position: Position[snx.NXsample, RunType],
) -> DetectorLtotal[RunType]:
    """Return moderator-to-detector flight path lengths for BEER McStas data."""
    source_to_sample = sc.norm(sample_position - source_position)
    sample_to_detector = sc.norm(da.coords['position'] - sample_position)
    return source_to_sample + sample_to_detector


mcstas_providers = (
    load_beer_mcstas_provider,
    load_beer_mcstas_monitor_provider,
    mcstas_source_position,
    mcstas_sample_position,
    mcstas_choppers,
    mcstas_detector_ltotal,
)
