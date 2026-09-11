# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Adapters for FREIA McStas files.

McStas conventions and the fixed WFM simulation configuration belong here.
The rest of the workflow uses the standard ESSreduce domain types.
"""

import re
from pathlib import Path
from typing import TypedDict

import mcstastox
import numpy as np
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import (
    DiskChoppers,
    EmptyDetector,
    Filename,
    NeXusDetectorName,
    Position,
    RawDetector,
    RunType,
)
from ess.reduce.unwrap import PulsePeriod
from scippneutron.chopper import DiskChopper


class _ChopperParameters(TypedDict):
    frequency: float
    position: list[float]
    open: list[float]
    close: list[float]


# WFM settings for FREIA_surface_test.instr with res=0.02, stopWBC=stopWFM=0.
# Frequencies are in Hz, global positions in m, and slit angles in degrees.
# Slit angles use ScippNeutron's convention with zero phase and beam position.
_WFM_PARAMETERS: dict[str, _ChopperParameters] = {
    'WBC1': {
        'frequency': 14.0,
        'position': [-0.025536188608227407, -0.2143487326684282, 6.313874089082927],
        'open': [265.75340934340136],
        'close': [347.41993036564133],
    },
    'PSC1': {
        'frequency': 56.0,
        'position': [-0.04279334439590373, -0.2487827747282733, 7.3252399717676475],
        'open': [
            225.05060997711988,
            174.31325763665376,
            125.39199315800286,
            79.4636849757332,
            35.19478493987566,
            353.0332202961111,
            314.3279223973939,
        ],
        'close': [
            231.97085763665373,
            183.04959315800286,
            136.45713088537676,
            92.08261269250694,
            49.820457639881454,
            369.57011076447253,
            331.2358038278435,
        ],
    },
    'PSC2': {
        'frequency': 56.0,
        'position': [-0.05036143156257555, -0.2612121196231304, 7.6903035721702],
        'open': [
            218.75947574117998,
            165.5769221153046,
            114.32685543062891,
            66.18060316860294,
            19.799339992501075,
            335.62596717152013,
            295.0046293340228,
        ],
        'close': [
            225.67972340071387,
            174.31325763665373,
            125.39199315800283,
            78.79953088537671,
            34.42501269250694,
            352.16285763988157,
            311.9125107644725,
        ],
    },
    'PSC3': {
        'frequency': 56.0,
        'position': [-0.06728900511986453, -0.2850923564964583, 8.391692529125612],
        'open': [
            206.1409753631455,
            148.79069468752678,
            93.06600550591139,
            40.84355449212529,
            350.3878524351604,
            302.33399651829416,
            256.56765150000723,
        ],
        'close': [
            218.89724809831387,
            162.22530499276786,
            108.60576573022513,
            56.985424074290464,
            368.2383919327576,
            321.81186800372,
            277.5953889804347,
        ],
    },
    'WBC2': {
        'frequency': 14.0,
        'position': [-0.08633018876666028, -0.3139402382145939, 9.238986706984262],
        'open': [227.07624477736135],
        'close': [339.9669648456413],
    },
    'PSC4': {
        'frequency': 42.0,
        'position': [-0.10516748527471297, -0.35051394569767463, 10.313196928583936],
        'open': [
            218.21716131667478,
            166.56540575149816,
            115.57883431227881,
            68.09433093440703,
            21.893063791537656,
            337.8911186406503,
            297.67374389476004,
        ],
        'close': [
            238.93286862631388,
            186.29390878964273,
            136.42670903352186,
            88.51839823856325,
            43.20268716658999,
            360.04414290207035,
            317.96091970176843,
        ],
    },
    'PSC5': {
        'frequency': 28.0,
        'position': [-0.11514600144160746, -0.5309902203230933, 15.613984633755468],
        'open': [
            220.38174306267877,
            168.49262474947486,
            118.50890129461459,
            72.09339945600789,
            28.148411442764086,
            344.6532571208708,
            303.481198960639,
        ],
        'close': [
            251.37265421671387,
            197.68679894272015,
            146.514171366115,
            97.48011281665286,
            50.54011861520438,
            366.1021937591697,
            323.6450492619195,
        ],
    },
    'WBC3': {
        'frequency': 14.0,
        'position': [-0.10215007498652816, -0.5905364416738501, 17.362922984132187],
        'open': [129.22970188564136],
        'close': [305.77460188564135],
    },
}


def wfm_choppers() -> DiskChoppers[RunType]:
    """Construct the fixed WFM cascade used by the initial FREIA simulations.

    Returns three bandwidth disks and five pulse-shaping/frame-overlap disks,
    each of the latter with seven openings. Settings are independent of the
    input file; a fresh set of disks is returned on each call.

    Positions are in the simulation's global frame, with the source at the origin.
    The analytical cascade projects them onto the z axis, approximating the
    curved guide and finite beam width by a central ray.
    """
    return DiskChoppers[RunType](
        {
            name: DiskChopper(
                frequency=sc.scalar(parameters['frequency'], unit='Hz'),
                beam_position=sc.scalar(0.0, unit='deg'),
                phase=sc.scalar(0.0, unit='deg'),
                axle_position=sc.vector(parameters['position'], unit='m'),
                slit_begin=sc.array(
                    dims=['cutout'], values=parameters['open'], unit='deg'
                ),
                slit_end=sc.array(
                    dims=['cutout'], values=parameters['close'], unit='deg'
                ),
            )
            for name, parameters in _WFM_PARAMETERS.items()
        }
    )


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _open_mcstas(filename: str | Path) -> mcstastox.Read:
    filename = Path(filename)
    return mcstastox.Read(filename.parent, filename.name)


def load_mcstas(
    filename: str | Path,
    detector_name: str = 'Multiblade',
    *,
    pulse_period: sc.Variable | None = None,
) -> sc.DataArray:
    """Load weighted events and pixel geometry from the final FREIA detector.

    Only the selected component is read. The expected output is the
    ``mantid banana`` event list, including ``p``, ``t``, ``id`` and pixel
    geometry. Histogram-only and upstream debug monitors are not substitutes.
    Empty detector pixels are retained, and weighted-event variances are ``p**2``.
    Arrival times are split into ``event_time_zero`` and ``event_time_offset``
    using ``pulse_period``, which defaults to the ESS period of 1/14 s.
    """
    if pulse_period is None:
        pulse_period = sc.scalar(1 / 14, unit='s')
    with _open_mcstas(filename) as data:
        return _load_events(data, detector_name, pulse_period)


def _load_events(
    data: mcstastox.Read,
    detector_name: str,
    pulse_period: sc.Variable,
    geometry: sc.DataArray | None = None,
) -> sc.DataArray:
    if detector_name not in data.get_components_with_ids():
        raise ValueError(
            f'No Mantid detector events with pixel IDs found for {detector_name!r}. '
            'Enable "mantid banana ... list all neutrons" in the simulation.'
        )
    values = data.get_event_data(
        variables=['p', 't', 'id'], component_name=detector_name, filter_zeros=True
    )
    if geometry is None:
        geometry = _detector_geometry(data, detector_name)
    pixel_ids = geometry.coords['pixel_id']
    if not np.isin(values['id'], pixel_ids.values).all():
        raise ValueError('Detector events contain pixel IDs absent from the pixel map.')
    time = sc.array(dims=['event'], values=values['t'], unit='s')
    offset = time % pulse_period.to(unit=time.unit)
    events = sc.DataArray(
        sc.array(
            dims=['event'],
            values=values['p'],
            variances=values['p'] ** 2,
            unit='counts',
        ),
        coords={
            'pixel_id': sc.array(
                dims=['event'], values=values['id'], dtype='int64', unit=None
            ),
            'event_time_offset': offset,
            'event_time_zero': sc.datetime(0, unit='ns')
            + (time - offset).to(unit='ns', dtype='int64'),
        },
    ).group(pixel_ids)
    return events.assign_coords(geometry.coords)


def load_mcstas_provider(
    filename: Filename[RunType],
    detector_name: NeXusDetectorName,
    geometry: EmptyDetector[RunType],
    pulse_period: PulsePeriod,
) -> RawDetector[RunType]:
    """Provide final-detector events, reusing the workflow's pixel geometry."""
    with _open_mcstas(filename) as data:
        return RawDetector[RunType](
            _load_events(data, detector_name, pulse_period, geometry)
        )


def _component_position(filename, component):
    with _open_mcstas(filename) as data:
        return sc.vector(data.get_global_component_coordinates(component), unit='m')


def mcstas_source_position(
    filename: Filename[RunType],
) -> Position[snx.NXsource, RunType]:
    """Load the moderator position from the simulation geometry."""
    return Position[snx.NXsource, RunType](_component_position(filename, 'Source'))


def mcstas_sample_position(
    filename: Filename[RunType],
) -> Position[snx.NXsample, RunType]:
    """Load the sample position from the simulation geometry."""
    return Position[snx.NXsample, RunType](_component_position(filename, 'Arm_Sample'))


def _histogram_axis(histogram, axis, unit):
    label = _text(histogram.attrs[f'{axis}label'])
    name = re.sub(r'[^a-zA-Z]', '_', label)
    # McStas encodes the units in the axis label, e.g., "y [m]".
    axis_unit = label[label.index('[') + 1 : label.index(']')]
    return (
        sc.array(
            dims=['pixel'], values=histogram[name][:], unit=axis_unit, dtype='float64'
        )
        .to(unit=unit)
        .values
    )


def _detector_geometry(data, detector_name) -> sc.DataArray:
    output = data.file_object.get_output_entry(detector_name)
    if 'BINS' in output:
        # Keep the file's ID order, including non-contiguous or permuted IDs.
        local = data.get_component_local(detector_name)
        position = data.get_component_global(detector_name)
        pixel_ids = np.asarray(
            data.file_object.get_pixels_entry(detector_name), dtype='int64'
        ).ravel()
    else:
        # Histogram axes also describe geometry; no intensities are read.
        geometry = data.file_object.get_geometry_dict(detector_name)
        if geometry['shape'] != 'banana':
            raise ValueError(f'Expected banana geometry for {detector_name!r}.')
        histogram = data.file_object.get_info_entry(detector_name)
        angle, height = np.meshgrid(
            _histogram_axis(histogram, 'x', 'rad'),
            _histogram_axis(histogram, 'y', 'm'),
        )
        radius = geometry['radius']
        local = np.column_stack(
            (
                (radius * np.sin(angle)).ravel(),
                height.ravel(),
                (radius * np.cos(angle)).ravel(),
            )
        )
        pixel_ids = np.arange(len(local))
        position = data.transform(local, detector_name)
    return sc.DataArray(
        sc.zeros(dims=['pixel_id'], shape=[len(local)], unit='counts'),
        coords={
            'pixel_id': sc.array(dims=['pixel_id'], values=pixel_ids, unit=None),
            'position': sc.vectors(dims=['pixel_id'], values=position, unit='m'),
            'longitude': sc.array(
                dims=['pixel_id'],
                values=np.rad2deg(np.arctan2(local[:, 0], local[:, 2])),
                unit='deg',
            ),
            'height': sc.array(dims=['pixel_id'], values=local[:, 1], unit='m'),
        },
    )


def mcstas_detector_geometry(
    filename: Filename[RunType], detector_name: NeXusDetectorName
) -> EmptyDetector[RunType]:
    """Provide detector geometry to the generic flight-path calculation.

    Works with Mantid pixel maps or banana histogram axes without reading events
    or intensities. The selected component's name and axes are read from metadata.
    """
    with _open_mcstas(filename) as data:
        return EmptyDetector[RunType](_detector_geometry(data, detector_name))


providers = (
    wfm_choppers,
    load_mcstas_provider,
    mcstas_source_position,
    mcstas_sample_position,
    mcstas_detector_geometry,
)
