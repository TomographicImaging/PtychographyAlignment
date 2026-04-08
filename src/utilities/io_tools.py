import h5py
import numpy as np
from typing import Dict, Tuple, List

def load_data(nxsfileName, data_key= '/entry1/tomo_entry/data/data_unwrapped', 
              angle_key = '/entry1/tomo_entry/data/rotation_angle', probe_key = '/entry1/pty_entry/probe/', 
              y_idx=[0,None,1], x_idx=[0,None,1], angle_idx=[0,None,1],
              ax_order = ['angle','y','x']):


    data_key_name = 'data'
    with h5py.File(nxsfileName,'r') as data_file: 
        data_shape = data_file[str(data_key)].shape
        print('raw data shape:', data_shape)

        slicing = {'angle':angle_idx,
                   'y': y_idx, 
                   'x': x_idx}
        
        index_to_ax= {idx: name for idx, name in enumerate(ax_order)}
        ax_to_index = {name: idx for idx, name in enumerate(ax_order)}

        # projections=np.array(data_file[str(data_key)][angle_idx[0]:angle_idx[1]:angle_idx[2],y_idx[0]:y_idx[1]:1,x_idx[0]:x_idx[1]:1]) #*(-1)
        projections=np.array(data_file[str(data_key)][slicing[index_to_ax[0]][0]:slicing[index_to_ax[0]][1]:slicing[index_to_ax[0]][2],
                                                      slicing[index_to_ax[1]][0]:slicing[index_to_ax[1]][1]:slicing[index_to_ax[1]][2],
                                                      slicing[index_to_ax[2]][0]:slicing[index_to_ax[2]][1]:slicing[index_to_ax[2]][2]]) #*(-1)
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

        projections = projections.transpose((ax_to_index['y'], ax_to_index['x'], ax_to_index['angle'])) # transpose to shape [Ny, Nx, Nangles]

    return projections, angles_rad, probes


def print_keys(name, obj):
    print(name, type(obj))