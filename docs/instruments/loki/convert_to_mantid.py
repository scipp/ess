import scipp as sc
import scippneutron as scn
import numpy as np
import h5py as h5
from shutil import copyfile
import mantid.simpleapi as mantid
import sys


def fix_nexus_mantid(infile, outfile):
    """
    Additional tweaks for loading into Mantid
    """

    copyfile(infile, outfile)
    with h5.File(outfile, 'r+') as f:

        f['entry/instrument/name'] = 'LARMOR'

        group_entry = f['entry']
        nx_class = group_entry.create_group('sample')
        nx_class.attrs["NX_class"] = 'NXsample'

        f['entry/larmor_detector_events'] = f['entry/instrument/larmor_detector/larmor_detector_events']
        del f['entry/instrument/larmor_detector/larmor_detector_events']

        f['entry/instrument/larmor_detector/monitor_1'] = f['entry/instrument/monitor_1']
        f['entry/instrument/larmor_detector/monitor_2'] = f['entry/instrument/monitor_2']

        del f['entry/instrument/monitor_1']
        del f['entry/instrument/monitor_2']


        group = f['entry/instrument']
        for monitor_name in filter(lambda k: k.startswith('monitor'), group):
            monitor_group = group[monitor_name]
            monitor_event_group = monitor_group[f'{monitor_name}_events']
            for key in list(monitor_event_group):
                monitor_group[key] = monitor_event_group.pop(key)
            del monitor_group[f'{monitor_name}_events']

def fix_nexus_scipp(infile, outfile):
    """
    Currently there are some tweaks to make file working in scipp. This function won't be necessary in the future
    """

    copyfile(infile, outfile)
    with h5.File(outfile, 'r+') as f:

        group = f['entry/instrument']
        for monitor_name in filter(lambda k: k.startswith('monitor'), group):
            monitor_group = group[monitor_name]
            monitor_event_group = monitor_group[f'{monitor_name}_events']
            for key in list(monitor_event_group):
                monitor_group[key] = monitor_event_group.pop(key)
            del monitor_group[f'{monitor_name}_events']
if __name__ == "__main__":
    """
    Script for converting to nexus Mantid files. Separate file for data and monitors are create. 
    "_scipp" file is required by scipp but it doesn't load in Mantid
    Note: *_monitors.nxs file is saved to the path defined by Mantid Workbench
    It can be change from "Manage Directories" menu in Mantid Workbench
    
    Usage: python convert_to_mantid.py 60378-2022-02-28_2215.nxs
    """
    input_file = sys.argv[1]
    fixed_file = f'{input_file[:-4]}_scipp.nxs'
    fixed_mantid_file = f'{input_file[:-4]}_mantid.nxs'
    fix_nexus_scipp(input_file, fixed_file)
    fix_nexus_mantid(input_file, fixed_mantid_file)
    data = scn.load_nexus(data_file=fixed_file)


    no_events = len(data.bins.constituents['data'].values)
    start_tof = data.coords['tof'][0].values
    end_tof = data.coords['tof'][-1].values
    nbins = 100
    tof_edges = sc.linspace(dim='tof', start=start_tof, stop=end_tof, num=nbins, unit='ns')
    histogrammed = sc.histogram(data, bins=tof_edges)

    monitors = {
        'sample': {'incident': data.attrs["monitor_1"].value,
                       'transmission': data.attrs["monitor_2"].value}
    }

    monitors['sample']['incident'].coords['position'] = sc.vector(value=np.array([0, 0, 0.0]), unit=sc.units.m)
    monitors['sample']['incident'].coords['source_position'] = sc.vector(value=np.array([0, 0, -25.3]),
                                                                             unit=sc.units.m)
    monitors['sample']['transmission'].coords['position'] = sc.vector(value=np.array([0, 0, 0.0]), unit=sc.units.m)
    monitors['sample']['transmission'].coords['source_position'] = sc.vector(value=np.array([0, 0, -25.3]),
                                                                                 unit=sc.units.m)

    #Using same tof_edges as for histogramming data
    histogrammed_monitor1 = sc.histogram(monitors['sample']['incident'], bins=tof_edges)
    histogrammed_monitor2 = sc.histogram(monitors['sample']['transmission'], bins=tof_edges)

    monitor1_mantid = scn.to_mantid(histogrammed_monitor1, dim='tof')
    monitor2_mantid = scn.to_mantid(histogrammed_monitor2, dim='tof')
    monitor_mantid_file = f'{input_file[:-4]}_monitors.nxs'
    mantid.SaveNexus(monitor1_mantid, monitor_mantid_file)
    mantid.SaveNexus(monitor2_mantid, monitor_mantid_file, Append=True)

