from utilities.Unwrap import Unwrap
from viewer.OpenViewer import OpenViewer
import numpy 

def test_unwrap_pollen_Volpe_data():
    from io_module.Imports import ImportData
    from config.paths import pollen_Volpe_filepath, pollen_Volpe_data_key, pollen_Volpe_angle_key
    data = ImportData(pollen_Volpe_filepath, data_key=pollen_Volpe_data_key, angle_key = pollen_Volpe_angle_key)
    projections_raw= data.get_projections_raw()
    OpenViewer(projections_raw)
    Unwrap(projections_raw)
    OpenViewer(projections_raw)

def test_unwrap_TestData():
    from test_utils.TestData import TestData
    data = TestData().data
    data1 = data
    data2 = data
    numpy.set_printoptions(suppress=True)
    OpenViewer(data)
    Unwrap(data, parallel=True)
    OpenViewer(data)
    Unwrap(data1, parallel=False, sliced = True)
    OpenViewer(data1)
    Unwrap(data2, parallel=False, sliced = False)
    OpenViewer(data2)

