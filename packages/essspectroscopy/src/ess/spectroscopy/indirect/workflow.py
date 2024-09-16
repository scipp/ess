from __future__ import annotations

from loguru import logger
from scipp import Variable

from ..types import NeXusFileName


def _load_all(group, obj_type):
    """Helper to find and load all subgroups of a specific scippnexus type"""
    return {name: obj[...] for name, obj in group[obj_type].items()}


def _load_named(group, obj_type, names):
    """Helper to find and load all subgroups of a specific scippnexus type with group name in an allowed set"""
    return {name: obj[...] for name, obj in group[obj_type].items() if name in names}


def ess_source_frequency():
    """Helper to create an input for a sciline workflow, returns the ESS source frequency of 14 Hz"""
    from scipp import scalar

    return scalar(14.0, unit='Hz')


def ess_source_period():
    """Helper to create an input for a sciline workflow, returns the ESS source period of 1/(14 Hz)"""
    return (1 / ess_source_frequency()).to(unit='ns')


def ess_source_delay():
    """Helper to create an input for a sciline workflow, returns per-wavelength source delays of 0 s"""
    from scipp import array

    return array(values=[0, 0.0], dims=['wavelength'], unit='sec', dtype='float64')


def ess_source_duration():
    """Helper to create an input for a sciline workflow, returns source pulse duration of 3 msec"""
    from scipp import scalar

    return scalar(3.0, unit='msec').to(unit='sec')


def ess_source_velocities():
    """Helper to create an input for a sciline workflow, returns per-wavelength source velocity limits

    Notes
    -----
    The chosen limits are not based on any properties of the source, but rather entirely on the equivalent
    energy range, which is chosen to be ~53 micro electron volts to 530 milli electron volts.
    This energy range should be sufficient for all intended incident energies of the ESS spectrometer suite,
    but may not be sufficient to capture spurious energies that pass through the real instruments.

    Returns
    -------
    :
        A 1-D scipp Variable with values=[100, 10000] m/s
    """
    from scipp import array

    return array(values=[100, 1e4], dims=['wavelength'], unit='m/s')


def convert_simulated_time_to_frame_time(data):
    """Helper to make McStas simulated event data look more like real data

    McStas has the ability to track the time-of-flight from source to detector for every probabilistic neutron ray.
    This is very helpful, but unfortunately real instrument at ESS are not able to record the same information
    due to how the timing and data collection systems work.

    Real neutron events will record their event_time_zero most-recent-pulse reference time, and their
    event_time_offset detection time relative to that reference time. These two values added together give
    a real wall time; and information about the primary spectrometer is necessary to find any time-of-flight

    This function takes event data with per-event coordinate event_time_offset (actually McStas time-of-flight)
    and creates a new coordinate frame_time that is the time-of-flight modulo the source repetition period.

    Notes
    -----
    If the input data has realistic event_time_offset values, this function should produce frame_time data
    which is identical, and will therefore only increase the amount of data stored per event.

    Returns
    -------
    :
        A copy of the data with extra per-event coordinate frame_time
    """
    graph = {
        'frame_time': lambda event_time_offset: event_time_offset % ess_source_period()
    }
    return data.transform_coords(
        'frame_time', graph=graph, rename_dims=False, keep_intermediate=False
    )


def analyzer_per_detector(analyzers: list[str], triplets: list[str]) -> dict[str, str]:
    """Find the right analyzer name for each detector

    Notes
    -----
    Depends heavily on the names of components being preceded by an in-instrument index,
    and the analyzer and detector components being separated in index by 2. If either condition changes
    this function will need to be modified.

    Parameters
    ----------
    analyzers: list[str]
        The names of analyzer components, typically generated from the keys of a dict
    triplets: dict
        The names of triplet detector components, typically generated from the keys of a dict

    Returns
    -------
    :
        A dictionary with detector name keys and their associated analyzer name values
    """

    # TODO Update this function if the NeXus group naming changes, or components are added/removed.
    def correct_index(d, a):
        detector_index = int(d.split('_', 1)[0])
        analyzer_index = detector_index - 2
        return a.startswith(str(analyzer_index))

    return {d: [x for x in analyzers if correct_index(d, x)][0] for d in triplets}


def detector_per_pixel(triplets: dict) -> dict[int, str]:
    """Find the right detector name for every pixel index

    Parameters
    ----------
    triplets: dict[str, scipp.DataGroup]
        A mapping of detector component name to a loaded scippnexus.NXdetector group, with 'data' group member
        that has a 'detector_number' coordinate

    Returns
    -------
    :
        The mapping of detector_number to detector component name
    """
    return {
        i: name
        for name, det in triplets.items()
        for i in det['data'].coords['detector_number'].values.flatten()
    }


def combine_analyzers(analyzers: dict, triplets: dict):
    """Combine needed analyzer properties into a single array, duplicating information, to have per-pixel data

    BIFROST has 45 analyzers and 45 triplet detectors, each with some number of pixels, N.
    Calculations for the properties of neutrons which make it to each detector pixel need per-pixel data about
    the analyzer associated with that pixel. This function collects the required data and combines it into a
    single array.

    Analyzer information required to determine the secondary-spectrometer neutron properties are the center-of-mass
    position of the analyzer, the orientation of the analyzer, and the lattice spacing separating crystal planes
    in the analyzer.

    Notes
    -----
    Since there are N pixels per detector, the returned array is strictly N-times larger than necessary,
    but the optimization to use only minimal information is left for the future

    Parameters
    ----------
    analyzers: dict[str, scipp.DataGroup]
        Maps analyzer component name to loaded scippnexus.NXcrystal group
    triplets: dict[str, scipp.DataGroup]
        Maps detector component name to loaded scippnexus.NXdetector group

    Returns
    -------
    :
        A single array with 'event_id' pixel dimension and the per-pixel analyzer information
    """
    from scipp import Dataset, array, concat
    from scippnexus import compute_positions

    def analyzer_extract(obj):
        obj = compute_positions(obj, store_transform='transform')
        return Dataset(data={k: obj[k] for k in ('position', 'transform', 'd_spacing')})

    extracted = {k: analyzer_extract(v) for k, v in analyzers.items()}
    d2a = analyzer_per_detector(list(analyzers.keys()), list(triplets.keys()))
    p2d = detector_per_pixel(triplets)

    p2a = {k: extracted[d2a[v]] for k, v in p2d.items()}
    pixels = sorted(p2a)
    data = concat([p2a[p] for p in pixels], dim='event_id')
    data['event_id'] = array(values=pixels, dims=['event_id'], unit=None)
    return data


def combine_detectors(triplets: dict):
    """Combine needed detector properties into a single array

    BIFROST has 45 analyzers and 45 triplet detectors, each with some number of pixels, N.
    Calculations for the properties of neutrons which make it to each detector pixel need per-pixel data.
    This function collects the required data and combines it into a single array.

    Detector information required to determine the secondary-spectrometer neutron properties are the center-of-mass
    position of each pixel.

    Parameters
    ----------
    triplets: dict[str, scipp.DataGroup]
        Maps detector component name to loaded scippnexus.NXdetector group

    Returns
    -------
    :
        A single array with 'event_id' pixel dimension and the per-pixel center of mass position
    """
    from scipp import Dataset, concat, sort

    def extract(obj):
        pixels = obj['data'].coords['detector_number']
        midpoints = obj['data'].coords['position']
        return Dataset(data={'event_id': pixels, 'position': midpoints})

    data = concat([extract(v) for v in triplets.values()], dim='arm')
    data = Dataset({k: v.flatten(to='event_id') for k, v in data.items()})
    return sort(data, data['event_id'].data)


def find_sample_detector_flight_time(sample, analyzers, detector_positions):
    """Use sciline to find the sample to detector flight time per detector pixel"""
    import numpy as np
    from sciline import Pipeline

    from ..types import (
        AnalyzerOrientation,
        AnalyzerPosition,
        DetectorPosition,
        ReciprocalLatticeSpacing,
        SampleDetectorFlightTime,
        SamplePosition,
    )
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
    """Extract and combine the events from loaded scippneutron.NXdetector groups

    Parameters
    ----------
    triplets:
        An iterable container of loaded NXdetector groups, each with a 'data' member which contains the
        pixel data -- possibly multiple detector-specific (but consistent) dimensions -- with a coordinate
        identifying the 'detector_number'

    Returns
    -------
    :
       The events from each triplet concatenated and sorted by the 'detector_number'
    """
    from scipp import concat, sort

    events = concat([x['data'] for x in triplets], dim='arm').flatten(to='event_id')
    events = sort(events, events.coords['detector_number'])
    return events


def get_sample_events(triplet_events, sample_detector_flight_times):
    """Copy the triplet events structure and offset its frame_time event coordinate to give frame_time at the sample"""
    events = triplet_events.copy()
    for coord in ('position', 'x_pixel_offset', 'y_pixel_offset'):
        del events.coords[coord]
    events.bins.coords['frame_time'] -= sample_detector_flight_times.to(unit='ns')
    events.bins.coords['frame_time'] %= ess_source_period()
    return events


def get_unwrapped_events(
    filename, source_name, sample_name, sample_events, focus_components
):
    """Use a sciline pipeline to shift frame_time at sample events to time-since-producing-pulse events at sample"""
    from sciline import Pipeline

    from ..types import (
        FocusComponentNames,
        NeXusFileName,
        PrimarySpectrometerObject,
        SampleFrameTime,
        SampleName,
        SampleTime,
        SourceDelay,
        SourceDuration,
        SourceFrequency,
        SourceName,
        SourceVelocities,
    )
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
    """Get the data of the named monitor component, converting frame_time to nanoseconds to match event_time_offset

    Parameters
    ----------
    monitors:
        A dictionary mapping monitor component name to loaded scippneutron.NXmonitor groups
    monitor_component:
        The name of the monitor component to access
    collapse: bool
        For some simulated experiments, a parameter was scanned which should not be treated as separate
        time points. When provied True, these points are integrated over the 'time' dimension.

    Returns
    -------
    :
        Monitor data with frame_time converted to nanoseconds to match the timescale used for events
    """
    normalization = monitors[monitor_component]['data']
    if collapse:
        # This is very specialized to how the simulated scans are done, it needs to be generalized
        normalization = normalization.sum(dim='time')
    # rescale the frame_time axis. Why does it need to be done this way?
    return normalization.transform_coords(
        ['frame_time'], graph={'frame_time': lambda t: t.to(unit='nanosecond')}
    )


def get_energy_axes(ki_params, kf_params):
    """Use a sciline pipeline to extract incident_energy, final_energy, and energy_transfer

    Parameters
    ----------
    ki_params:
        A dictionary of parameters needed by the incident-spectrometer sciline pipeline
    kf_params:
        A dictionary of parameters needed by the secondary-spectrometer sciline pipeline

    Returns
    -------
    ei:
        The incident energy
    en:
        The energy transfer
    ef:
        The final energy
    """
    from sciline import Pipeline

    from ..types import (
        EnergyTransfer,
        FinalEnergy,
        FinalWavenumber,
        IncidentEnergy,
        IncidentWavenumber,
    )
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
    """Use a sciline pipeline to extract momentum transfer in the lab and sample-table coordinate systems

    Parameters
    ----------
    ki_params:
        A dictionary of parameters needed by the incident-spectrometer sciline pipeline
    kf_params:
        A dictionary of parameters needed by the secondary-spectrometer sciline pipeline
    events:
        The event data to which the calculated momentum components are appended
    a3:
        The scalar value of the sample rotation angle describing the events

    Returns
    -------
    :
        The event data with the two horizontal plane components of the momentum transfer added in the
        laboratory coordinate system (independent of a3) and the sample-table coordinate system (rotated by a3 around y)
        These new coordinates are named 'lab_momentum_x', 'lab_momentum_z', 'table_momentum_x' and 'table_momentum_z'
    """
    from sciline import Pipeline

    from ..types import (
        LabMomentumTransfer,
        LabMomentumTransferX,
        LabMomentumTransferZ,
        SampleTableAngle,
        TableMomentumTransfer,
        TableMomentumTransferX,
        TableMomentumTransferZ,
    )
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
    :
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


def load_everything(filename: NeXusFileName, named_components: dict[str, str]):
    """Load all needed information from the named NeXus HDF5 file

    Parameters
    ----------
    filename:
        The name of the file to load data from, must have both and 'instrument' and 'parameters' group under 'entry'
    named_components:
        The file-specific names of (at least) the source and sample group names under 'entry/instrument'

    Returns
    -------
    sample:
        The loaded sample component group
    triplets:
        All scippnexus.NXdetector groups under 'entry/instrument'
    analyzers:
        All scippnexus.NXcrystal groups under 'entry/instrument'
    choppers:
        All scippnexus.NXdisk_chopper groups under 'entry/instrument'
    monitors:
        All scippnexus.NXmonitor groups under 'entry/instrument'
    logs:
        The scippnexus.NXlog groups named 'a3' and 'a4' under 'entry/parameters'
    """
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
        triplets = _load_all(group, snx.NXdetector)
        analyzers = _load_all(group, snx.NXcrystal)
        choppers = _load_all(group, snx.NXdisk_chopper)
        monitors = _load_all(group, snx.NXmonitor)

        # this is very BIFROST simulation specific -- can it be less so?
        # TODO use the _sample_ orientation itself to define a3 (and goniometer angles?)
        logs = _load_named(data['entry/parameters'], snx.NXlog, ('a3', 'a4'))

    return sample, triplets, analyzers, choppers, monitors, logs


def one_setting(
    sample, triplet_events, analyzers, norm_monitor, filename, names, warn_about_a3=True
):
    """Calculate the event properties for a single (a3, a4) setting"""
    detector_positions = triplet_events.coords['position']
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
    filename: NeXusFileName,
    named_components: dict[str, str],
    is_simulated: bool = False,
):
    """Load data from a NeXus file and perform (a3, a4) independent calculations

    Parameters
    ----------
    filename:
        The file which contains the data to load
    named_components:
        The file-specific names of (at least) the source, sample and normalization monitor group names
        under 'entry/instrument'
    is_simulated:
        A flag to indicate if the file comes from a McStas simulation, such that the event_time_offset
        needs to be modified to look like real data.

    Returns
    -------
    sample:
        The sample group loaded from the NeXus file
    analyzers:
        A single array with all analyzer information needed to calculate secondary-spectrometer parameters
    triplet_events:
        A single array with the events and detector-pixel information needed to calculate parameters
    norm_monitor:
        The normalization monitor data with frame_time converted to nanoseconds
    logs:
        A dictionary of the 'a3' and 'a4' logs from the 'entry/parameter' group in the NeXus file
    """
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
    triplet_events = get_triplet_events(triplets.values())

    norm_monitor = get_normalization_monitor(monitors, named_components['monitor'])
    return sample, analyzers, triplet_events, norm_monitor, logs


def component_names(
    source_component: str | None = None,
    sample_component: str | None = None,
    focus_components: list[str] | None = None,
    monitor_component: str | None = None,
    is_simulated: bool = False,
):
    """Return a dictionary mapping component type to component name

    Parameters
    ----------
    source_component: str
        The user-provided source component name, should exist at 'entry/instrument/{source_component}' in the datafile
    sample_component: str
        The user-provided sample component name, should exist at 'entry/instrument/{sample_component}' in the datafile
    focus_components: list[str]
        The user-provided set of component names defining the time-focus position, each should exist under
        'entry/instrument' in the datafile
    monitor_component: str
        The user-provided normalization monitor component name, should exist at 'entry/instrument/{monitor_component}'
    is_simulated: bool
        If true, user-provided names will be augmented with the McStas component names for the specific types.

    Returns
    -------
    :
        A dictionary mapping component type name to group name
    """
    from ..types import FocusComponentName

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
    """Load a BIFROST data file and convert to S(Q,E) in the sample-table coordinate system

    Parameters
    ----------
    filename:
        The name of the NeXus file to load
    source_component:
        The group name under 'entry/instrument' in the NeXus file containing source information
    sample_component:
        The group name under 'entry/instrument' in the NeXus file containing sample information
    focus_components:
        The group name or group names under 'entry/instrument' in the NeXus file which define the focus-time
    monitor_component:
        The group name under 'entry/instrument' in the NeXus file containing normalization monitor information
    is_simulated:
        Whether the NeXus file comes from a McStas simulation, in which case default component names are set
        if not provided and the data is modified to look like real data

    Returns
    -------
    A dictionary of data from the workflow, concatenated along a 'setting' dimension corresponding to separate
    (a3, a4) grouped data. The entries in the dictionary may not all be useful, and are subject to pruning as
    experience is gained with the workflow.
    """
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


def bifrost_single(
    filename: NeXusFileName,
    source_component: str | None = None,
    sample_component: str | None = None,
    focus_components: list[str] | None = None,
    monitor_component: str | None = None,
    is_simulated: bool = False,
    extras: bool = False,
):
    """Load a BIFROST data file and convert to S(Q,E) in the laboratory coordinate system

    Parameters
    ----------
    filename:
        The name of the NeXus file to load
    source_component:
        The group name under 'entry/instrument' in the NeXus file containing source information
    sample_component:
        The group name under 'entry/instrument' in the NeXus file containing sample information
    focus_components:
        The group name or group names under 'entry/instrument' in the NeXus file which define the focus-time
    monitor_component:
        The group name under 'entry/instrument' in the NeXus file containing normalization monitor information
    is_simulated:
        Whether the NeXus file comes from a McStas simulation, in which case default component names are set
        if not provided and the data is modified to look like real data
    extras:
        If true, the loaded sample group and 'a3' and 'a4' logs will be returned in the dictionary

    Returns
    -------
    A dictionary of data from the workflow. The entries in the dictionary may not all be useful,
    and are subject to pruning as experience is gained with the workflow.
    """
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
