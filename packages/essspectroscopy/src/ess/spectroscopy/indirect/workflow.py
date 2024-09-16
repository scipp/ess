from __future__ import annotations

import sys

from loguru import logger

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "{elapsed} {message}"
)
logger.configure(extra={"ip": "", "user": ""})  # Default values
logger.remove()
logger.add(sys.stderr, format=logger_format)

from ..types import *


def load_all(group, obj_type):
    return {name: obj[...] for name, obj in group[obj_type].items()}


def load_named(group, obj_type, names):
    return {name: obj[...] for name, obj in group[obj_type].items() if name in names}


def ess_source_frequency():
    from scipp import scalar

    return scalar(14.0, unit='Hz')


def ess_source_period():
    return (1 / ess_source_frequency()).to(unit='ns')


def ess_source_delay():
    from scipp import array

    return array(values=[0, 0.0], dims=['wavelength'], unit='sec', dtype='float64')


def ess_source_duration():
    from scipp import scalar

    return scalar(3.0, unit='msec').to(unit='sec')


def ess_source_velocities():
    from scipp import array

    return array(
        values=[100, 1e4], dims=['wavelength'], unit='m/s'
    )  # ~53 ueV to 530 meV


def convert_simulated_time_to_frame_time(data):
    graph = {
        'frame_time': lambda event_time_offset: event_time_offset % ess_source_period()
    }
    return data.transform_coords(
        'frame_time', graph=graph, rename_dims=False, keep_intermediate=False
    )


def analyzer_per_detector(analyzers, triplets):
    """Find the right analyzer name for each detector"""

    # TODO Update this function if the NeXus group naming changes, or components are added/removed.
    def correct_index(d, a):
        detector_index = int(d.split('_', 1)[0])
        analyzer_index = detector_index - 2
        return a.startswith(str(analyzer_index))

    return {
        d: [x for x in analyzers.keys() if correct_index(d, x)][0]
        for d in list(triplets.keys())
    }


def detector_per_pixel(triplets):
    """Find the right detector name for every pixel index"""
    return {
        i: name
        for name, det in triplets.items()
        for i in det['data'].coords['detector_number'].values.flatten()
    }


def combine_analyzers(analyzers, triplets):
    """Combine needed analyzer properties into a single array, duplicating information, to have per-pixel data"""
    from scipp import Dataset, array, concat
    from scippnexus import compute_positions

    def analyzer_extract(obj):
        obj = compute_positions(obj, store_transform='transform')
        return Dataset(data={k: obj[k] for k in ('position', 'transform', 'd_spacing')})

    extracted = {k: analyzer_extract(v) for k, v in analyzers.items()}
    d2a = analyzer_per_detector(analyzers, triplets)
    p2d = detector_per_pixel(triplets)

    p2a = {k: extracted[d2a[v]] for k, v in p2d.items()}
    pixels = sorted(p2a)
    data = concat([p2a[p] for p in pixels], dim='event_id')
    data['event_id'] = array(values=pixels, dims=['event_id'], unit=None)
    return data


def combine_detectors(triplets):
    from scipp import Dataset, concat, sort

    def extract(obj):
        pixels = obj['data'].coords['detector_number']
        midpoints = obj['data'].coords['position']
        return Dataset(data={'event_id': pixels, 'position': midpoints})

    data = concat([extract(v) for v in triplets.values()], dim='arm')
    data = Dataset({k: v.flatten(to='event_id') for k, v in data.items()})
    return sort(data, data['event_id'].data)


def find_sample_detector_flight_time(sample, analyzers, detector_positions):
    import numpy as np
    from sciline import Pipeline

    from .kf import providers as kf_providers

    params = {
        SamplePosition: sample['position'],
        AnalyzerPosition: analyzers['position'].data,
        DetectorPosition: detector_positions,  # detectors['position'].data,
        AnalyzerOrientation: analyzers['transform'].data,
        ReciprocalLatticeSpacing: 2 * np.pi / analyzers['d_spacing'].data,
    }
    return params, Pipeline(kf_providers, params=params).get(
        SampleDetectorFlightTime
    ).compute().to(unit='ms')


def get_triplet_events(triplets):
    from scipp import concat, sort

    events = concat([x['data'] for x in triplets.values()], dim='arm').flatten(
        to='event_id'
    )
    events = sort(events, events.coords['detector_number'])
    return events


def get_sample_events(triplet_events, sample_detector_flight_times):
    events = triplet_events.copy()
    for coord in ('position', 'x_pixel_offset', 'y_pixel_offset'):
        del events.coords[coord]
    events.bins.coords['frame_time'] -= sample_detector_flight_times.to(unit='ns')
    events.bins.coords['frame_time'] %= ess_source_period()
    return events


def get_unwrapped_events(
    filename, source_name, sample_name, sample_events, focus_components
):
    from sciline import Pipeline

    from .ki import providers as ki_providers

    params = {
        NeXusFileName: filename,
        SampleName: sample_name,
        SourceName: source_name,
        SourceDelay: ess_source_delay(),
        SourceDuration: ess_source_duration(),
        SourceFrequency: ess_source_frequency(),
        SourceVelocities: ess_source_velocities(),
        SampleFrameTime: sample_events.data.bins.coords['frame_time'],
        FocusComponentNames: focus_components,
    }
    pipeline = Pipeline(ki_providers, params=params)
    primary = pipeline.get(PrimarySpectrometerObject).compute()

    events = sample_events.copy()
    events.bins.coords['frame_time'] = pipeline.get(SampleTime).compute()
    return params, events, primary


def get_normalization_monitor(monitors, monitor_component, collapse: bool = False):
    normalization = monitors[monitor_component]['data']
    if collapse:
        # This is very specialized to how the simulated scans are done, it needs to be generalized
        normalization = normalization.sum(dim='time')
    # rescale the frame_time axis. Why does it need to be done this way?
    return normalization.transform_coords(
        ['frame_time'], graph={'frame_time': lambda t: t.to(unit='nanosecond')}
    )


def get_energy_axes(ki_params, kf_params):
    from sciline import Pipeline

    from .conservation import providers

    params = {}
    params.update(ki_params)
    params.update(kf_params)
    pipeline = Pipeline(providers, params=params)
    pipeline[IncidentWavenumber] = pipeline.get(IncidentWavenumber).compute()
    pipeline[FinalWavenumber] = pipeline.get(FinalWavenumber).compute()
    ei = pipeline.get(IncidentEnergy).compute()
    en = pipeline.get(EnergyTransfer).compute()
    ef = pipeline.get(FinalEnergy).compute()
    return ei, en, ef


def add_momentum_axes(ki_params, kf_params, events, a3: Variable):
    from sciline import Pipeline

    from .conservation import providers

    if a3.size != 1:
        raise ValueError(f'Expected a3 to have 1-entry, not {a3.size}')

    params = {}
    # First we must add the lab momentum vector, since it is not a3 dependent
    params.update(ki_params)
    params.update(kf_params)
    params[SampleTableAngle] = a3

    pipeline = Pipeline(providers, params=params)
    pipeline[LabMomentumTransfer] = pipeline.get(LabMomentumTransfer).compute()

    events.bins.coords['lab_momentum_x'] = pipeline.get(LabMomentumTransferX).compute()
    # events.bins.coords['lab_momentum_y'] = pipeline.get(LabMomentumTransferY).compute()
    events.bins.coords['lab_momentum_z'] = pipeline.get(LabMomentumTransferZ).compute()

    pipeline[TableMomentumTransfer] = pipeline.get(TableMomentumTransfer).compute()
    events.bins.coords['table_momentum_x'] = (
        pipeline.get(TableMomentumTransferX).compute().transpose(events.dims)
    )
    # events.bins.coords['table_momentum_y'] = pipeline.get(TableMomentumTransferY).compute().transpose(events.dims)
    events.bins.coords['table_momentum_z'] = (
        pipeline.get(TableMomentumTransferZ).compute().transpose(events.dims)
    )
    return events


def split(
    triplets,
    analyzers,
    monitors,
    logs,
    a3_name: str | None = None,
    a4_name: str | None = None,
):
    """Use the (a3, a4) logged value pairs to split triplet, analyzer, and monitor data into single-setting sets

    Parameters
    ----------
    triplets: DataArray
        The triplet positions _should_ change with a4 (if simulated, this is certainly not implemented correctly yet)
        The event data they contain depends on (a3, a4) [plus any other time-dependent parameter] so must be split.
    analyzers: Dataset?
        The analyzer position _should_ change with a4 (if simulated, this is certainly not implemented correctly yet)
    monitors: DataArray
        The histogram (simulated, or real current beam monitor) or event (real fission monitor, etc.?) data is
        time-dependent and used to normalize the detector data. It must therefore be split into (a3, a4) sets
    logs: DataGroup
        Entries built from NXlogs in the real or simulated instrument
    a3_name: str
        The name of the sample table angle log entry in `logs` -- 'a3' for simulated data
    a4_name: str
        The name of the detector tank angle log entry in `logs` -- 'a4' for simulated data

    Returns
    -------
    A list[[triplet, analyzer, monitor]] of individual (a3, a4) setting(s)
    """
    from scipp import lookup

    from ..utils import is_in_coords, split_setting

    if a3_name is None:
        a3_name = 'a3'
    if a4_name is None:
        a4_name = 'a4'

    if a3_name not in logs or a4_name not in logs:
        logger.warning('Missing a3 or a4, so split performed')
        return [[triplets, analyzers, monitors]]

    a3 = lookup(logs[a3_name]['value'], 'time')
    a4 = lookup(logs[a4_name]['value'], 'time')

    event_graph = {
        'a3': lambda event_time_zero: a3[event_time_zero],
        'a4': lambda event_time_zero: a4[event_time_zero],
    }
    histogram_graph = {'a3': lambda time: a3[time], 'a4': lambda time: a4[time]}

    def do_split(x, time_name):
        graph = event_graph if 'event_time_zero' == time_name else histogram_graph
        if is_in_coords(x, time_name):
            x = x.transform_coords(('a3', 'a4'), graph=graph)
            if x.bins is not None:
                x = x.group('a3', 'a4')
        return x

    vals = [
        do_split(x, t)
        for x, t in (
            (triplets, 'event_time_zero'),
            (analyzers, 'time'),
            (monitors, 'time'),
        )
    ]

    # FIXME this only works because v.sizes['a4'] is always 1 at the moment
    vals = [
        v.flatten(['a3', 'a4'], to='time') if 'a3' in v.dims and 'a4' in v.dims else v
        for v in vals
    ]

    n_time = [v.sizes['time'] for v in vals if 'time' in v.dims]
    if len(n_time):
        assert all(n == n_time[0] for n in n_time)
        n_time = n_time[0]
        vals = [
            [v['time', i] if 'time' in v.dims else v for v in vals]
            for i in range(n_time)
        ]
    else:
        vals = [vals]

    return vals


def load_everything(filename, named_components):
    import scippnexus as snx

    source_component = named_components['source']
    sample_component = named_components['sample']
    with snx.File(filename) as data:
        group = data['entry/instrument']
        if source_component not in group:
            raise ValueError(
                f'Missing source component {source_component} for path-length calculations'
            )
        if sample_component not in group:
            raise ValueError(
                f'Missing sample component {sample_component} for origin identification'
            )
        sample = snx.compute_positions(
            group[sample_component][...], store_transform='transform'
        )
        triplets = load_all(group, snx.NXdetector)
        analyzers = load_all(group, snx.NXcrystal)
        choppers = load_all(group, snx.NXdisk_chopper)
        monitors = load_all(group, snx.NXmonitor)

        # this is very BIFROST simulation specific -- can it be less so?
        # TODO use the _sample_ orientation itself to define a3 (and goniometer angles?)
        logs = load_named(data['entry/parameters'], snx.NXlog, ('a3', 'a4'))

    return sample, triplets, analyzers, choppers, monitors, logs


def one_setting(
    sample, triplet_events, analyzers, norm_monitor, filename, names, warn_about_a3=True
):
    detector_positions = triplet_events.coords[
        'position'
    ]  # this is the same as detectors['position'].data!
    kf_params, sample_detector_flight_time = find_sample_detector_flight_time(
        sample, analyzers, detector_positions
    )
    sample_events = get_sample_events(triplet_events, sample_detector_flight_time)
    ki_params, unwrapped_events, primary = get_unwrapped_events(
        filename, names['source'], names['sample'], sample_events, names['focus']
    )
    ei, en, ef = get_energy_axes(ki_params, kf_params)

    energy_events = sample_events.copy()
    energy_events.bins.coords['energy_transfer'] = en.to(unit='meV')
    energy_events.bins.coords['incident_energy'] = ei
    energy_events.coords['final_energy'] = ef

    if 'a3' in triplet_events.coords:
        # this _should_ be one (a3, a4) setting, with a single a3 value on triple_events (and norm_monitor)
        a3 = triplet_events.coords['a3']
    else:
        from scipp import scalar

        if warn_about_a3:
            logger.warning("No a3 present in setting, assuming 0 a3")
        a3 = scalar(0, unit='deg')

    energy_momentum_events = add_momentum_axes(ki_params, kf_params, energy_events, a3)

    return {
        'triplet_events': triplet_events,
        'sample_events': sample_events,
        'unwrapped_events': unwrapped_events,
        'norm_monitor': norm_monitor,
        'energy_events': energy_events,
        'sample_detector_flight_time': sample_detector_flight_time,
        'analyzers': analyzers,
        'energy_momentum_events': energy_momentum_events,
        # 'detectors': detectors,
        # 'monitors': monitors,
        # 'triplets': triplets,
    }


def load_precompute(
    filename: NeXusFileName, named_components, is_simulated: bool = False
):
    import scippnexus as snx

    sample, triplets, analyzers, choppers, monitors, logs = load_everything(
        filename, named_components
    )

    if is_simulated:
        for name in triplets:
            triplets[name]['data'] = convert_simulated_time_to_frame_time(
                triplets[name]['data']
            )

    for name in triplets:
        triplets[name] = snx.compute_positions(
            triplets[name], store_transform='transform'
        )

    analyzers = combine_analyzers(analyzers, triplets)
    # detectors = combine_detectors(triplets)
    triplet_events = get_triplet_events(triplets)

    norm_monitor = get_normalization_monitor(monitors, named_components['monitor'])
    return sample, analyzers, triplet_events, norm_monitor, logs


def component_names(
    source_component: str | None = None,
    sample_component: str | None = None,
    focus_components: list[str] | None = None,
    monitor_component: str | None = None,
    is_simulated: bool = False,
):
    names = {
        'source': source_component,
        'sample': sample_component,
        'focus': focus_components,
        'monitor': monitor_component,
    }
    if is_simulated:
        sim_components = {
            'source': '001_ESS_source',
            'sample': '114_sample_stack',
            'monitor': '110_frame_3',
            'focus': [
                FocusComponentName('005_PulseShapingChopper'),
                FocusComponentName('006_PulseShapingChopper2'),
            ],
        }
        for k, v in sim_components.items():
            if names[k] is None:
                names[k] = v
    return names


def bifrost(
    filename: NeXusFileName,
    source_component: str | None = None,
    sample_component: str | None = None,
    focus_components: list[str] | None = None,
    monitor_component: str | None = None,
    is_simulated: bool = False,
):
    import scipp as sc
    from tqdm import tqdm

    named_components = component_names(
        source_component,
        sample_component,
        focus_components,
        monitor_component,
        is_simulated,
    )
    sample, analyzers, triplet_events, norm_monitor, logs = load_precompute(
        filename, named_components, is_simulated
    )
    settings = split(triplet_events, analyzers, norm_monitor, logs)
    data = [
        one_setting(
            sample,
            one_triplet_events,
            one_analyzers,
            one_monitor,
            filename,
            named_components,
        )
        for one_triplet_events, one_analyzers, one_monitor in tqdm(
            settings, desc='(a3, a4) settings'
        )
    ]
    return {k: sc.concat([d[k] for d in data], 'setting') for k in data[0]}


# def bifrost_dask(filename: NeXusFileName,
#             source_component: str | None = None,
#             sample_component: str | None = None,
#             focus_components: list[str] | None = None,
#             monitor_component: str | None = None,
#             is_simulated: bool=False):
#     import scipp as sc
#     from tqdm.dask import TqdmCallback
#     import dask
#     named_components = component_names(
#         source_component, sample_component, focus_components, monitor_component, is_simulated
#     )
#     sample, analyzers, triplet_events, norm_monitor, logs = load_precompute(filename, named_components, is_simulated)
#     settings = split(triplet_events, analyzers, norm_monitor, logs)
#
#     def concat_one(parts):
#         return sc.concat(parts, dim='setting')
#
#     do_one = dask.delayed(one_setting)
#     concat_one = dask.delayed(concat_one)
#
#     # This fails because ... something ... can't be pickled -- scipp things?
#     data = [do_one(sample, ev, an, mn, filename, named_components) for ev, an, mn in settings]
#     with TqdmCallback(desc="(a3, a4) setting"):
#         data = dask.compute(*data)
#
#     keys = list(data[0].keys())
#     with TqdmCallback(desc="Combine settings"):
#         futures = dask.persist(*[concat_one([d[k] for d in data]) for k in keys])
#         return {k: v for k, v in zip(keys, dask.compute(*futures))}


def bifrost_single(
    filename: NeXusFileName,
    source_component: str | None = None,
    sample_component: str | None = None,
    focus_components: list[str] | None = None,
    monitor_component: str | None = None,
    is_simulated: bool = False,
    extras: bool = False,
):
    named_components = component_names(
        source_component,
        sample_component,
        focus_components,
        monitor_component,
        is_simulated,
    )
    sample, analyzers, triplet_events, norm_monitor, logs = load_precompute(
        filename, named_components, is_simulated
    )

    data = one_setting(
        sample,
        triplet_events,
        analyzers,
        norm_monitor,
        filename,
        named_components,
        warn_about_a3=False,
    )

    if extras:
        data['sample'] = sample
        data['logs'] = logs

    return data
