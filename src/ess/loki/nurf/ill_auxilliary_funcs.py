import errno
import os
import h5py
import numpy as np
import copy 

def complete_fname(scan_numbers):
    """Converts a list of input numbers to a filename uses at ILL.

    Parameters
    ----------
    scan_numbers: list of int or a single int
        List of filenumbers or one filenumnber.

    Returns:
    ----------
    flist_num: list of str or  one  str
        List of filenames following ILL style or string following ILL style.

    """
    if isinstance(scan_numbers, int):
        flist_num = f"{str(scan_numbers).zfill(6)}.nxs"

    if isinstance(scan_numbers, list):
        # convert a list of input numbers to real filename
        flist_num = [str(i).zfill(6) + ".nxs" for i in scan_numbers]

    return flist_num

# This cell contains function to convert ILL raw data to the forseen Loki file format

def loki_file_creator(loki_path, loki_filename):
    """
    This function creates a dummy loki_basic.nxs that has the expected file structure for ESS Loki.
    Afterwards, I will append the nurf data and rename the file.
    """
    with h5py.File(os.path.join(loki_path,loki_filename), 'w') as hf:
        nxentry = hf.create_group("entry")
        nxentry.attrs['NX_class'] = 'NXentry'
        nxinstrument=nxentry.create_group("instrument") 
        nxinstrument.attrs['NX_class'] = 'NXinstrument'


def load_one_spectro_file(file_handle, path_rawdata):
    """
    This function loads one .nxs file containing spectroscopy data (fluo, uv). Data is stored in multiple np.ndarrays.

    In:
     file_handle: file_handle is a file number, one element is expected otherwise an error is raised
                type: list

    path_rawdata: Path to the raw data
                type: str

    Out:
        data: contains all relevant HDF5 entries and their content for the Nurf project (keys and values)
            type: dict

    """

    # create path to file, convert file_number to string
    file_path_spectro = os.path.join(path_rawdata, file_handle + '.nxs')

    # check if file exists
    if not os.path.isfile(file_path_spectro):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                file_path_spectro)

    # we are ready to load the data set
    # open the .nxs file and read the values
    with h5py.File(file_path_spectro, "r") as f:
        # access nurf sub-group
        nurf_group = '/entry0/D22/nurf/'

        # access keys in sub-group
        nurf_keys = list(f[nurf_group].keys())
        # how many keys exist
        len_nurf_keys = len(nurf_keys)

        # print(nurf_keys)
        # print(len_nurf_keys)

        # this returns a list with HDF5datasets (I think)
        # data_spectro_file=list(f[nurf_group].values())

        # data_spectro_file=f[nurf_group].values()
        # data_spectro_file=f[nurf_group]

        # extract all data of the nurf subgroup and store it in a new dict

        # initialise an empty dict
        data = {}

        for key in f[nurf_group].keys():
            # print(key)
            # this is how I get string giving the full path to this dataset
            path_dataset = f[nurf_group][key].name
            # print(path_dataset)
            # print(f[nurf_group][key].name) #this prints the full path to the dataset
            # print(type(f[path_dataset][:])) #this gives me access to the content of the data set, I could use : or () inside [], out: np.ndarray

            # This gives a dict with full path name as dict entry followed by value. No, I don't want this, but good to know.
            # data[f[nurf_group][key].name]=f[path_dataset][:]

            # This gives a dict where the keys corresponds to the key names of the h5 file.
            data[key] = f[path_dataset][:]

        # print(f[nurf_group].get('Fluo_spectra'))

        # print a hierachical view of the file (simple)
        # like this is would only go through subgroups
        # f[nurf_group].visititems(lambda x, y: print(x))

        # walk through the whole file and show the attributes (found on github as an explanation for this function)
        # def print_attrs(name, obj):
        #    print(name)
        #    for key, val in obj.attrs.items():
        #        print("{0}: {1}".format(key, val))
        # f[nurf_group].visititems(print_attrs)

        # print(data_spectro_file)

    # file_handle is returned as np.ndarray and its an np.array, elements correspond to row indices
    return data


def nurf_file_creator(loki_file, path_to_loki_file, data):
    """
    Appends NUrF group to LOKI NeXus file for ESS

    Args:
        loki_file (str): filename of NeXus file for Loki
        path_to_loki_file (str): File path where the NeXus file for LOKI is stored
        data (dict): Dictionary with dummy data for Nurf
    """

    # change directory where the loki.nxs is located
    os.chdir(path_to_loki_file)

    # open the file and append
    with h5py.File(loki_file, 'a') as hf:
                
        #comment on names
        #UV/FL_Background is the dark
        #UV/FL_Intensity0 is the reference
        #UV/FL_Spectra is the sample
        
        # image_key: number of frames (nFrames) given indirectly as part of the shape of the arrays 
        # TODO: keep in mind what happens if multiple dark or reference frames are taken
        
        # remove axis=2 of length one from this array
        data['UV_spectra']=np.squeeze(data['UV_spectra'], axis=2)  #this removes the third axis in this array, TODO: needs later to be verified with real data from hardware
        
        # assemble all spectra in one variable
        uv_all_data=np.row_stack((data['UV_spectra'],data['UV_background'], data['UV_intensity0']))
      
        dummy_vec=np.full(np.shape(uv_all_data)[0], False) 

        #create boolean masks for data, dark, reference
        uv_nb_spectra=np.shape(data['UV_spectra'])[0]

        # data mask, copy and replace first entries 
        uv_data_mask=copy.copy(dummy_vec)
        uv_data_mask[0:uv_nb_spectra]=True

        # dark mask
        # find out how many darks exist
        # TODO: Is there always a background or do we need to catch this case if there isn't?
        if data['UV_background'].ndim==1:
            uv_nb_darks=1
        else: 
            uv_nb_darks=np.shape(data['UV_background'])[1]  #TODO: needs to be verified with real data from Judith's setup

        uv_dark_mask=copy.copy(dummy_vec)
        uv_dark_mask[uv_nb_spectra:uv_nb_spectra+uv_nb_darks]=True

        # reference 
        # how many references where taken? 
        if data['UV_intensity0'].ndim==1:
            uv_nb_ref=1
        else:
            uv_nb_ref=np.shape(data['UV_intensity0'])[1]  #TODO: needs to be verified with real data from Judith's setup
        
        # reference mask, copy and replace first entries 
        uv_ref_mask=copy.copy(dummy_vec)
        uv_ref_mask[uv_nb_spectra+uv_nb_darks:uv_nb_spectra+uv_nb_darks+uv_nb_ref]=True

        
        
        # UV subgroup
        grp_uv = hf.create_group("/entry/instrument/uv")
        grp_uv.attrs["NX_class"] = 'NXdata'
        
        # uv spectra
        uv_signal_data=grp_uv.create_dataset('data', data=uv_all_data, dtype=np.float32)
        uv_signal_data.attrs['long name']= 'all_data'
        uv_signal_data.attrs['units']= 'counts'
        grp_uv.attrs['signal']= 'data'  #indicate that the main signal is data 
        grp_uv.attrs['axes']= [ "spectrum", "wavelength" ] #time is here the first axis, i.e axis=0, wavelength is axis=1
        
        # define the AXISNAME_indices
        grp_uv.attrs['time_indices'] = 0
        grp_uv.attrs['integration_time_indices'] = 0
        grp_uv.attrs['wavelength_indices'] = 1

        # introducing a key that is interpretable for sample, dark, and reference 
        grp_uv.attrs['is_data_indices'] = 0
        grp_uv.attrs['is_dark_indices'] = 0
        grp_uv.attrs['is_reference_indices'] = 0  

        grp_uv.create_dataset('is_data', data=uv_data_mask, dtype=bool)
        grp_uv.create_dataset('is_dark', data=uv_dark_mask, dtype=bool)
        grp_uv.create_dataset('is_reference', data=uv_ref_mask, dtype=bool)

        # uv_time
        # dummy timestamps for uv_time
        # TODO: Codes will have to change later for the real hardware.
        uv_time = np.empty(np.shape(uv_all_data)[0], dtype='datetime64[us]')  
        for i in range(0, np.shape(uv_time)[0]):
            uv_time[i]=np.datetime64('now')
    
        # see https://stackoverflow.com/questions/23570632/store-datetimes-in-hdf5-with-h5py 
        # suggested work around because h5py does not support time types
        uv_time_data=grp_uv.create_dataset('time', data=uv_time.view('<i8'), dtype='<i8')
        uv_time_data.attrs['unit'] = 'us'
        # to read
        #print(uv_time_data[:].view('<M8[us]'))
        # TODO: Do we need here an attribute for the unit?
       
        # uv_wavelength
        uv_wavelength_data=grp_uv.create_dataset('wavelength', data=data['UV_wavelength'], dtype=np.float32)
        
        uv_wavelength_data.attrs['units'] = 'nm'  # TODO: unit to be verified
        uv_wavelength_data.attrs['long name'] = 'wavelength'
                
                
        # creating for each spectrum, even dark and background, the integration time
        # TODO: needs to be verified with real hardware
        uv_inttime=np.full(np.shape(uv_all_data)[0],data['UV_IntegrationTime'])
        
         # uv_integration_time
        uv_inttime_data=grp_uv.create_dataset("integration_time",data=uv_inttime, dtype=np.int32)
                   
        uv_inttime_data.attrs['long_name'] = 'integration_time'
        uv_inttime_data.attrs['units'] = 'us'  # TODO: unit to be verified, currently in micro-seconds

        # Fluorescence subgroup
        grp_fluo = hf.create_group("/entry/instrument/fluorescence")
        grp_fluo.attrs["NX_class"] = 'NXdata'
        
        # currenty real fluo data is often messed up (i.e. empty spectra inbetween real ones)
        # remove third axis of length one
        data['Fluo_spectra']=np.squeeze(data['Fluo_spectra'], axis=2)

        # Something is not okay with the real Fluo_intensity0 data from ILL. It contains only one 0, at least in my file from the ILL beamtime. 
        # Also, some fluo spcetra in between are just dummy ones. There were acquisition problems.
        # This code block can later be adjusted
        try:
            assert (np.shape(data['Fluo_intensity0'])[0]!=1), 'Fluo_intensity0 contains only one value.'
        except AssertionError as error:
            data['Fluo_intensity0']=data['Fluo_intensity0'][0]*np.ones((np.shape(data['Fluo_background'])[0]))
           
        # assemble all fluo data    
        fluo_all_data=np.row_stack((data['Fluo_spectra'],data['Fluo_background'], data['Fluo_intensity0']))

        # dummy vec for mask
        dummy_vec=np.full(np.shape(fluo_all_data)[0], False) 

        # how many fluo data spectra
        fluo_nb_spectra=np.shape(data['Fluo_spectra'])[0]
        
        #create boolean masks for data, dark, reference
        # data mask, copy and replace first entries 
        fluo_data_mask=copy.copy(dummy_vec)
        fluo_data_mask[0:fluo_nb_spectra]=True

        #how many backgrounds?
        if data['Fluo_background'].ndim==1:
            fluo_nb_dark=1
        else:
            fluo_nb_dark=np.shape(data['Fluo_background'])[1] #TODO: needs to be verified with Judith's setup
        fluo_dark_mask=copy.copy(dummy_vec)
        fluo_dark_mask[fluo_nb_spectra:fluo_nb_spectra+fluo_nb_dark]=True

        #how many references? 
        if data['Fluo_intensity0'].ndim==1:
            fluo_nb_ref=1
        else:
            fluo_nb_ref=np.shape(data['Fluo_intensity0'])[1] #TODO: needs to be verified with Judith's setup

        fluo_ref_mask=copy.copy(dummy_vec)
        fluo_ref_mask[fluo_nb_spectra+fluo_nb_dark:fluo_nb_spectra+fluo_nb_dark+fluo_nb_ref]=True
                
        
        # maybe it does not matter at what monowavelength dark and reference are measured, but I have to add then dummy values
        # dummy value for monowavelength for dark:  NaN nm
        # dummy value for monowavelength for reference (until I know):  NaN nm
        # again the fluo data can be messed up here, example: 1 monowavelength, but 7 fluo spectra in  103418.nxs
        try:
            assert (np.shape(data['Fluo_monowavelengths'])[0]!=1), 'Fluo_monowavelengths contains only one value.'
        except AssertionError as error:
            data['Fluo_monowavelengths']=data['Fluo_monowavelengths'][0]*np.zeros((np.shape(data['Fluo_spectra'])[0]))
        
        #how many monowavelengths ?  
        nb_monowavelengths=data['Fluo_monowavelengths'].size
        
        # create a dummy vector for monowavelength 
        dummy_mwl=np.ones(dummy_vec.shape)*(-1)
        # I replace the first entries with the corect Fluo_monowavelengths
        dummy_mwl[0:nb_monowavelengths]=data['Fluo_monowavelengths']
        # now I need to add the monowavelengths for dark and reference, but they are not stored with the ILL data :(
        # I go for NaN
        # add monowavelengths for number of darks
        dummy_mwl[nb_monowavelengths:nb_monowavelengths+fluo_nb_dark]=np.nan
        # add monowavelengths for number of reference
        dummy_mwl[nb_monowavelengths+fluo_nb_dark:nb_monowavelengths+fluo_nb_dark+fluo_nb_ref]=np.nan
        
        data['Fluo_monowavelengths']=dummy_mwl

        # fluo spectra
        fluo_signal_data = grp_fluo.create_dataset('data',
                                               data=fluo_all_data, dtype=np.float32)
        fluo_signal_data.attrs['long name'] = 'all_data'
        fluo_signal_data.attrs['units'] = 'counts'
    

        grp_fluo.attrs['signal']= 'data'  #indicate that the main signal is data 
        grp_fluo.attrs['axes']= [ "spectrum",  "wavelength"]
        
        # define the AXISNAME_indices
        grp_fluo.attrs['time_indices'] = 0
        grp_fluo.attrs['integration_time_indices'] = 0
        grp_fluo.attrs['monowavelengths_indices'] = 0
        grp_fluo.attrs['wavelength_indices'] = 1
        

        grp_fluo.attrs['is_data_indices'] = 0
        grp_fluo.attrs['is_dark_indices'] = 0
        grp_fluo.attrs['is_reference_indices'] = 0  

        grp_fluo.create_dataset('is_data',data=fluo_data_mask,dtype=bool)
        grp_fluo.create_dataset('is_dark',data=fluo_dark_mask,dtype=bool)
        grp_fluo.create_dataset('is_reference',data=fluo_ref_mask,dtype=bool)

       
        # fluo_time
        # dummy timestamps for fluo_time
        # TODO: Codes will have to change later for the real hardware.
        fluo_time = np.empty(np.shape(fluo_all_data)[0], dtype='datetime64[us]')  
        for i in range(0, np.shape(fluo_time)[0]):
            fluo_time[i]=np.datetime64('now')
    
        # see https://stackoverflow.com/questions/23570632/store-datetimes-in-hdf5-with-h5py 
        # suggested work around because h5py does not support time types
        fluo_time_data=grp_fluo.create_dataset('time', data=fluo_time.view('<i8'), dtype='<i8')
        fluo_time_data.attrs['units'] = 'us'
        # to read
        #print(fluo_time_data[:].view('<M8[us]'))
        # TODO: Do we need here an attribute for the unit?
       
       
        # creating integration time for each fluo spectrum including dark and reference
        # TODO: needs to be verfied with hardware
        fluo_inttime=np.full(np.shape(fluo_all_data)[0],data['Fluo_IntegrationTime'])
       
        # fluo_integration_time
        fluo_inttime_data = grp_fluo.create_dataset('integration_time',
                                               data=fluo_inttime,
                                               dtype=np.float32)
        fluo_inttime_data.attrs['units'] = 'us'  # TODO: unit to be verified, currently micro-seconds
        fluo_inttime_data.attrs['long name'] = 'integration_time'


        # fluo_monowavelengths
        fluo_monowavelengths_data = grp_fluo.create_dataset('monowavelengths',
                                                       data=data[
                                                           'Fluo_monowavelengths'],
                                                       dtype=np.float32)
        fluo_monowavelengths_data.attrs['units'] = 'nm'  # TODO: unit to be verified
        fluo_monowavelengths_data.attrs['long name'] = 'monowavelengths'

        
        # fluo_wavelength
        fluo_wavelength_data= grp_fluo.create_dataset('wavelength',
                                                  data=data['Fluo_wavelength'],
                                                  dtype=np.float32)
        fluo_wavelength_data.attrs['units'] = 'nm'  # TODO: unit to be verified
        fluo_wavelength_data.attrs['long name'] = 'fluo_wavelength'

        # dummy groups, no information currently available
        #grp_sample_cell = hf.create_group("/entry/sample/sample_cell")
        #grp_sample_cell.attrs["NX_class"] = 'NXenvironment'
        #grp_sample_cell.create_dataset('description', data='NUrF sample cell')
        #grp_sample_cell.create_dataset('type', data='SQ1-ALL')

        #grp_pumps = hf.create_group("/entry/sample/hplc_pump")
        #grp_pumps.attrs["NX_class"] = 'NXenvironment'
        #grp_pumps.create_dataset("description", data='HPLC_pump')

        #no more valves
        #grp_valves = grp_nurf.create_group("Valves")
        #grp_valves.attrs["NX_class"] = 'NXenvironment'
        #grp_valves.create_dataset("description", data='Valves')

        #grp_densito = hf.create_group("/entry/instrument/densitometer")
        #grp_densito.attrs["NX_class"] = 'NXdetector'
        #grp_densito.create_dataset("description", data='Densitometer')

def convert_ill2loki(scan_numbers, path_to_loki_file, loki_file, path_rawdata, process_folder):
    """
    Convert ill data sets of files to new LoKI data structure.
    scan_numbers: list of filenumbers without trailing zero
    path_to_loki_file: path to LOKI_basics.nxs
    loki_file:  LOKI_Baiscs.nxs, dummy LOKI file
    path_rawdata: filepath/to/ILL/rawdata
    process_folder: filepath/to/folder/with/files/with/new/LOKI structure

    """

    # convert numbers into strings
    flist_num=[str(i).zfill(6) for i in scan_numbers]

    for i in flist_num:
        print(i, type(i))

        # create new LOKI_basics.nxs
        loki_file_creator(path_to_loki_file, loki_file)

        # load old ILL data and extract data
        data=load_one_spectro_file(i, path_rawdata)

        # put extracted data into the new LOKI file structure by appending to loki_file
        nurf_file_creator(loki_file, path_to_loki_file, data)

        # rename LOKI_basic files to new file name
        os.chdir(process_folder)
        final_fname=i+'.nxs'
        os.rename(loki_file, final_fname)