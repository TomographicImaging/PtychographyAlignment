from io_module.Imports import ImportData
from viewer.OpenViewer import OpenViewer
from config.paths import pollen_Volpe_filepath, pollen_Volpe_data_key, pollen_Volpe_angle_key
def test_open_viewer():
    """Tests that the viewer works correctly"""
    data = ImportData(pollen_Volpe_filepath, data_key= pollen_Volpe_data_key, angle_key = pollen_Volpe_angle_key)
    projections_raw= data.get_projections_raw()
    OpenViewer(projections_raw)
