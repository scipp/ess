from __future__ import annotations

from .conservation import energy
from ..types import *


def load_all(group, obj_type):
    return {name: obj[...] for name, obj in group[obj_type].items()}


def ess_source_frequency():
    from scipp import scalar
    return scalar(14.0, unit='Hz')


def ess_source_period():
    return (1 / ess_source_frequency()).to(unit='ns')


def ess_source_delay():
    from scipp import array
    return array(values=[0, 0.], dims=['wavelength'], unit='sec', dtype='float64')


def ess_source_duration():
    from scipp import scalar
    return scalar(3.0, unit='msec').to(unit='sec')


def ess_source_velocities():
    from scipp import array
    return array(values=[100, 1e4], dims=['wavelength'], unit='m/s')  # ~53 ueV to 530 meV


def convert_simulated_time_to_frame_time(data):
    graph = {'frame_time': lambda event_time_offset: event_time_offset % ess_source_period()}
    return data.transform_coords('frame_time', graph=graph, rename_dims=False, keep_intermediate=False)


def analyzer_per_detector(analyzers, triplets):
    """Find the right analyzer name for each detector"""
    # TODO Update this function if the NeXus group naming changes, or components are added/removed.
    def correct_index(d, a):
        detector_index = int(d.split('_', 1)[0])
        analyzer_index = detector_index - 2
        return a.startswith(str(analyzer_index))

    return {d: [x for x in analyzers.keys() if correct_index(d, x)][0] for d in list(triplets.keys())}


def detector_per_pixel(triplets):
    """Find the right detector name for every pixel index"""
    return {i: name for name, det in triplets.items() for i in det['data'].coords['detector_number'].values.flatten()}


def combine_analyzers(analyzers, triplets):
    """Combine needed analyzer properties into a single array, duplicating information, to have per-pixel data"""
    from scippnexus import compute_positions
    from scipp import Dataset, concat, array
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
        return Dataset(data={'event_id':  pixels, 'position': midpoints})

    data = concat([extract(v) for v in triplets.values()], dim='arm')
    data = Dataset({k: v.flatten(to='event_id') for k, v in data.items()})
    return sort(data, data['event_id'].data)


def find_sample_detector_flight_time(sample, analyzers, detectors):
    from sciline import Pipeline
    import numpy as np
    from .kf import providers as kf_providers
    params = {
        SamplePosition: sample['position'],
        AnalyzerPosition: analyzers['position'].data,
        DetectorPosition: detectors['position'].data,
        AnalyzerOrientation: analyzers['transform'].data,
        ReciprocalLatticeSpacing: 2 * np.pi / analyzers['d_spacing'].data,
    }
    return params, Pipeline(kf_providers, params=params).get(SampleDetectorFlightTime).compute().to(unit='ms')


def get_triplet_events(triplets):
    from scipp import concat, sort
    events = concat([x['data'] for x in triplets.values()], dim='arm').flatten(to='event_id')
    events = sort(events, events.coords['detector_number'])
    return events


def get_sample_events(triplet_events, sample_detector_flight_times):
    events = triplet_events.copy()
    for coord in ('position', 'x_pixel_offset', 'y_pixel_offset'):
        del events.coords[coord]
    events.bins.coords['frame_time'] -= sample_detector_flight_times.to(unit='ns')
    events.bins.coords['frame_time'] %= ess_source_period()
    return events


def get_unwrapped_events(filename, source_name, sample_name, sample_events, focus_components):
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
        FocusComponentNames: focus_components
    }
    pipeline = Pipeline(ki_providers, params=params)
    primary = pipeline.get(PrimarySpectrometerObject).compute()

    events = sample_events.copy()
    events.bins.coords['frame_time'] = pipeline.get(SampleTime).compute()
    return params, events, primary


def get_normalization_monitor(monitors, monitor_component):
    # This is very specialized to how the simulated scans are done, it needs to be generalized
    normalization = monitors[monitor_component]['data'].sum(dim='time')
    # rescale the frame_time axis. Why does it need to be done this way?
    return normalization.transform_coords(['frame_time'], graph={'frame_time': lambda t: t.to(unit='nanosecond')})


def get_energy_axes(ki_params, kf_params):
    from sciline import Pipeline
    from .conservation import providers
    params = {}
    params.update(ki_params)
    params.update(kf_params)
    pipeline = Pipeline(providers, params=params)
    ei = pipeline.get(IncidentEnergy).compute()
    en = pipeline.get(EnergyTransfer).compute()
    ef = pipeline.get(FinalEnergy).compute()
    return ei, en, ef

def bifrost(filename: NeXusFileName,
            source_component: str | None = None,
            sample_component: str | None = None,
            focus_components: list[str] | None = None,
            monitor_component: str | None = None,
            is_simulated: bool=False):
    import scippnexus as snx
    if source_component is None and is_simulated:
        source_component = '001_ESS_source'
    if sample_component is None and is_simulated:
        sample_component = '114_sample_stack'
    if focus_components is None and is_simulated:
        focus_components = [FocusComponentName('005_PulseShapingChopper'), FocusComponentName('006_PulseShapingChopper2')]
    if monitor_component is None and is_simulated:
        monitor_component = '110_frame_3'


    with snx.File(filename) as data:
        group = data['entry/instrument']
        if source_component not in group:
            raise ValueError(f'Missing source component {source_component} for path-length calculations')
        if sample_component not in group:
            raise ValueError(f'Missing sample component {sample_component} for origin identification')
        sample = snx.compute_positions(group[sample_component][...], store_transform='transform')
        triplets = load_all(group, snx.NXdetector)
        analyzers = load_all(group, snx.NXcrystal)
        choppers = load_all(group, snx.NXdisk_chopper)
        monitors = load_all(group, snx.NXmonitor)


    if is_simulated:
        for name in triplets:
            triplets[name]['data'] = convert_simulated_time_to_frame_time(triplets[name]['data'])

    for name in triplets:
        triplets[name] = snx.compute_positions(triplets[name], store_transform='transform')

    analyzers = combine_analyzers(analyzers, triplets)
    detectors = combine_detectors(triplets)

    kf_params, sample_detector_flight_time = find_sample_detector_flight_time(sample, analyzers, detectors)

    triplet_events = get_triplet_events(triplets)
    sample_events = get_sample_events(triplet_events, sample_detector_flight_time)

    ki_params, unwrapped_events, primary = get_unwrapped_events(filename, source_component, sample_component, sample_events, focus_components)

    norm_monitor = get_normalization_monitor(monitors, monitor_component)

    ei, en, ef = get_energy_axes(ki_params, kf_params)

    energy_events = sample_events.copy()
    energy_events.bins.coords['energy_transfer'] = en.to(unit='meV')
    energy_events.bins.coords['incident_energy'] = ei
    energy_events.coords['final_energy'] = ef

    #return triplet_events, sample_events, unwrapped_events, norm_monitor, energy_events, sample_detector_flight_time, analyzers, detectors
    return {
        'triplet_events': triplet_events,
        'sample_events': sample_events,
        'unwrapped_events': unwrapped_events,
        'norm_monitor': norm_monitor,
        'energy_events': energy_events,
        'sample_detector_flight_time': sample_detector_flight_time,
        'analyzers': analyzers,
        'detectors': detectors,
    }