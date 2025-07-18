from io_module.Imports import ImportData
from viewer.OpenViewer import OpenViewer

ptytomofile = 'C:/Users/zvm34551/Coding_environment/DATA/Ptychography/Experimental/pollen_Volpe/pty_tomo_NX.h5'

def test_open_viewer():
    """Tests that the viewer works correctly"""
    data = ImportData(ptytomofile, data_key='/entry1/tomo_entry/data/data', angle_key = '/entry1/tomo_entry/data/rotation_angle')
    projections_raw= data.get_projections_raw()
    OpenViewer(projections_raw)
