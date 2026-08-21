import h5py
import numpy as np

def load_data(nxsfileName, data_key= '/entry1/tomo_entry/data/data_unwrapped', angle_key = '/entry1/tomo_entry/data/rotation_angle', probe_key = '/entry1/pty_entry/probe/', 
              y_idx=[0,None], x_idx=[0,None], angle_idx=[0,None,1]):


    if nxsfileName.endswith('.h5') or nxsfileName.endswith('.nxs') or nxsfileName.endswith('.hdf5'):
        
        with h5py.File(nxsfileName,'r') as data_file: 
            data_shape = data_file[str(data_key)].shape
            print('raw data shape:', data_shape)


            projections=np.array(data_file[str(data_key)][angle_idx[0]:angle_idx[1]:angle_idx[2],y_idx[0]:y_idx[1]:1,x_idx[0]:x_idx[1]:1])#*(-1)
            np.nan_to_num(projections, copy=False)

            angles = np.array(data_file[angle_key][angle_idx[0]:angle_idx[1]:angle_idx[2]])
            #projections -= np.amin(projections)
            angles -= np.amin(angles)

            angles_rad = angles * np.pi/180.0

            #projections = np.pad(projections, ((0,0), (0,0), (50,50)), 'edge')

            probes = None
            if probe_key in data_file.keys():
                probes = np.array(data_file[probe_key+'modulus']) * np.exp(1j* np.array(data_file[probe_key+'phase']))

            print('projections shape:', projections.shape)
            print('angles shape:', angles_rad.shape)

    elif nxsfileName.endswith('.mat'):
        import scipy.io as sio
        data_file = sio.loadmat(nxsfileName)

        projections = np.array(data_file[str(data_key)][angle_idx[0]:angle_idx[1]:angle_idx[2],y_idx[0]:y_idx[1]:1,x_idx[0]:x_idx[1]:1])
        np.nan_to_num(projections, copy=False)

        angles = np.array(data_file[angle_key][angle_idx[0]:angle_idx[1]:angle_idx[2]])
        angles -= np.amin(angles)
        angles_rad = angles[0,:] * np.pi/180.0

        probes = None
        print('projections shape:', projections.shape)
        print('angles shape:', angles_rad.shape)
        

    else:
        raise ValueError('File format not supported. Please provide a .h5, .nxs, .hdf5 or .mat file.')
    return projections, angles_rad, probes