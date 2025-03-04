# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import re
from collections.abc import Generator

import scipp as sc
import scippnexus as snx

from ..types import (
    CrystalRotation,
    DetectorBankPrefix,
    DetectorIndex,
    DetectorName,
    FilePath,
    MaximumCounts,
    MaximumProbability,
    MaximumTimeOfArrival,
    McStasWeight2CountScaleFactor,
    MinimumTimeOfArrival,
    NMXDetectorMetadata,
    NMXExperimentMetadata,
    NMXRawDataMetadata,
    NMXRawEventCountsDataGroup,
    PixelIds,
    RawEventProbability,
)
from .xml import McStasInstrument, read_mcstas_geometry_xml


def detector_name_from_index(index: DetectorIndex) -> DetectorName:
    return f'nD_Mantid_{getattr(index, "value", index)}'


def load_event_data_bank_name(
    detector_name: DetectorName, file_path: FilePath
) -> DetectorBankPrefix:
    '''Finds the filename associated with a detector'''
    with snx.File(file_path) as file:
        description = file['entry1/instrument/description'][()]
        for bank_name, det_names in bank_names_to_detector_names(description).items():
            if detector_name in det_names:
                return DetectorBankPrefix(bank_name.partition('.')[0])
    raise KeyError(
        f"{DetectorBankPrefix.__name__} cannot be found for "
        f"{DetectorName.__name__} from the file {FilePath.__name__}"
    )


def _exclude_zero_events(data: sc.Variable) -> sc.Variable:
    """Exclude events with zero counts from the data.

    McStas can add an extra event line containing 0,0,0,0,0,0
    This line should not be included so we skip it.
    """
    if (data.values[0] == 0).all():
        data = data["event", 1:]
    else:
        data = data
    return data


def _wrap_raw_event_data(data: sc.Variable) -> RawEventProbability:
    data = data.rename_dims({'dim_0': 'event'})
    data = _exclude_zero_events(data)
    event_da = sc.DataArray(
        coords={
            'id': sc.array(
                dims=['event'],
                values=data['dim_1', 4].values,
                dtype='int64',
                unit=None,
            ),
            't': sc.array(dims=['event'], values=data['dim_1', 5].values, unit='s'),
        },
        data=sc.array(dims=['event'], values=data['dim_1', 0].values, unit='counts'),
    )
    return RawEventProbability(event_da)


def load_raw_event_data(
    file_path: FilePath, *, detector_name: DetectorName, bank_prefix: DetectorBankPrefix
) -> RawEventProbability:
    """Retrieve events from the nexus file.

    Parameters
    ----------
    file_path:
        Path to the nexus file
    detector_name:
        Name of the detector to load
    bank_prefix:
        Prefix identifying the event data array containing the events of the detector
        If None, the bank name is determined automatically from the detector name.

    """
    if bank_prefix is None:
        bank_prefix = load_event_data_bank_name(detector_name, file_path)
    bank_name = f'{bank_prefix}_dat_list_p_x_y_n_id_t'
    with snx.File(file_path, 'r') as f:
        root = f["entry1/data"]
        (bank_name,) = (name for name in root.keys() if bank_name in name)
        data = root[bank_name]["events"][()]
        return _wrap_raw_event_data(data)


def raw_event_data_chunk_generator(
    file_path: FilePath,
    *,
    detector_name: DetectorName,
    bank_prefix: DetectorBankPrefix | None = None,
    chunk_size: int = 0,  # Number of rows to read at a time
) -> Generator[RawEventProbability, None, None]:
    """Chunk events from the nexus file.

    Parameters
    ----------
    file_path:
        Path to the nexus file
    detector_name:
        Name of the detector to load
    pixel_ids:
        Pixel ids to generate the data array with the events
    chunk_size:
        Number of rows to read at a time.
        If 0, chunk slice is determined automatically by the ``iter_chunks``.
        Note that it only works if the dataset is already chunked.

    """
    if 0 < chunk_size < 10_000_000:
        import warnings

        warnings.warn(
            "The chunk size may be too small < 10_000_000.\n"
            "Consider increasing the chunk size for better performance.\n"
            "Hint: NMX typically expect ~10^8 bins as reduced data.",
            UserWarning,
            stacklevel=2,
        )

    # Find the data bank name associated with the detector
    bank_prefix = load_event_data_bank_name(
        detector_name=detector_name, file_path=file_path
    )
    bank_name = f'{bank_prefix}_dat_list_p_x_y_n_id_t'
    with snx.File(file_path, 'r') as f:
        root = f["entry1/data"]
        (bank_name,) = (name for name in root.keys() if bank_name in name)

    with snx.File(file_path, 'r') as f:
        root = f["entry1/data"]
        dset = root[bank_name]["events"]
        if chunk_size == 0:
            for data_slice in dset.dataset.iter_chunks():
                dim_0_slice, _ = data_slice  # dim_0_slice, dim_1_slice
                da = _wrap_raw_event_data(dset["dim_0", dim_0_slice])
                if da.sizes['event'] < 10_000_000:
                    import warnings

                    warnings.warn(
                        "The chunk size may be too small < 10_000_000.\n"
                        "Consider increasing the chunk size for better performance.\n"
                        "Hint: NMX typically expect ~10^8 bins as reduced data.",
                        UserWarning,
                        stacklevel=2,
                    )
                yield da

        else:
            num_events = dset.shape[0]
            for start in range(0, num_events, chunk_size):
                data = dset["dim_0", start : start + chunk_size]
                yield _wrap_raw_event_data(data)


def load_crystal_rotation(
    file_path: FilePath, instrument: McStasInstrument
) -> CrystalRotation:
    """Retrieve crystal rotation from the file.

    Raises
    ------
    KeyError
        If the crystal rotation is not found in the file.

    """
    with snx.File(file_path, 'r') as file:
        param_keys = tuple(f"entry1/simulation/Param/XtalPhi{key}" for key in "XYZ")
        if not all(key in file for key in param_keys):
            raise KeyError(
                f"Crystal rotations [{', '.join(param_keys)}] not found in file."
            )
        return CrystalRotation(
            sc.vector(
                value=[file[param_key][...] for param_key in param_keys],
                unit=instrument.simulation_settings.angle_unit,
            )
        )


def maximum_probability(da: RawEventProbability) -> MaximumProbability:
    """Find the maximum probability in the data."""
    return MaximumProbability(da.data.max())


def mcstas_weight_to_probability_scalefactor(
    max_counts: MaximumCounts, max_probability: MaximumProbability
) -> McStasWeight2CountScaleFactor:
    """Calculate the scale factor to convert McStas weights to counts.

    max_counts * (probabilities / max_probability)

    Parameters
    ----------
    max_counts:
        The maximum number of counts after scaling the event counts.

    scale_factor:
        The scale factor to convert McStas weights to counts

    """

    return McStasWeight2CountScaleFactor(
        sc.scalar(max_counts, unit="counts") / max_probability
    )


def bank_names_to_detector_names(description: str) -> dict[str, list[str]]:
    """Associates event data names with the names of the detectors
    where the events were detected"""

    detector_component_regex = (
        # Start of the detector component definition, contains the detector name.
        r'^COMPONENT (?P<detector_name>.*) = Monitor_nD\(\n'
        # Some uninteresting lines, we're looking for 'filename'.
        # Make sure no new component begins.
        r'(?:(?!COMPONENT)(?!filename)(?:.|\s))*'
        # The line that defines the filename of the file that stores the
        # events associated with the detector.
        r'(?:filename = \"(?P<bank_name>[^\"]*)\")?'
    )
    matches = re.finditer(detector_component_regex, description, re.MULTILINE)
    bank_names_to_detector_names = {}
    for m in matches:
        bank_names_to_detector_names.setdefault(
            # If filename was not set for the detector the filename for the
            # event data defaults to the name of the detector.
            m.group('bank_name') or m.group('detector_name'),
            [],
        ).append(m.group('detector_name'))
    return bank_names_to_detector_names


def load_experiment_metadata(
    instrument: McStasInstrument, crystal_rotation: CrystalRotation
) -> NMXExperimentMetadata:
    """Load the experiment metadata from the McStas file."""
    return NMXExperimentMetadata(
        sc.DataGroup(
            crystal_rotation=crystal_rotation, **instrument.experiment_metadata()
        )
    )


def load_detector_metadata(
    instrument: McStasInstrument, detector_name: DetectorName
) -> NMXDetectorMetadata:
    """Load the detector metadata from the McStas file."""
    return NMXDetectorMetadata(
        sc.DataGroup(**instrument.detector_metadata(detector_name))
    )


def load_mcstas(
    *,
    da: RawEventProbability,
    experiment_metadata: NMXExperimentMetadata,
    detector_metadata: NMXDetectorMetadata,
) -> NMXRawEventCountsDataGroup:
    return NMXRawEventCountsDataGroup(
        sc.DataGroup(weights=da, **experiment_metadata, **detector_metadata)
    )


def retrieve_pixel_ids(
    instrument: McStasInstrument, detector_name: DetectorName
) -> PixelIds:
    """Retrieve the pixel IDs for a given detector."""
    return PixelIds(instrument.pixel_ids(detector_name))


def retrieve_raw_data_metadata(
    min_toa: MinimumTimeOfArrival,
    max_toa: MaximumTimeOfArrival,
    max_probability: MaximumProbability,
) -> NMXRawDataMetadata:
    """Retrieve the metadata of the raw data."""
    return NMXRawDataMetadata(
        sc.DataGroup(
            min_toa=min_toa,
            max_toa=max_toa,
            max_probability=max_probability,
        )
    )


providers = (
    retrieve_raw_data_metadata,
    read_mcstas_geometry_xml,
    detector_name_from_index,
    load_event_data_bank_name,
    load_raw_event_data,
    maximum_probability,
    mcstas_weight_to_probability_scalefactor,
    retrieve_pixel_ids,
    load_crystal_rotation,
    load_mcstas,
    load_experiment_metadata,
    load_detector_metadata,
)
