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
    data1 = data.copy()
    data2 = data.copy()
    numpy.set_printoptions(suppress=True)
    OpenViewer(data)
    Unwrap(data, parallel=True)
    OpenViewer(data)
    Unwrap(data1, parallel=False, sliced = True)
    OpenViewer(data1)
    Unwrap(data2, parallel=False, sliced = False)
    OpenViewer(data2)

def test_parallel_jobs():
    from joblib import Parallel, delayed
    a = numpy.array([[0,0,0],[10,10,10],[20,20,20]])
    b = numpy.array([a.copy(), a.copy(), a.copy()])
    b_expected = numpy.array([2*a, 2*a, 2*a])

    def func(arr2d):
        arr2d += arr2d

    Parallel(n_jobs=-1, prefer="threads")(delayed(func)(arr) for arr in b)
    assert numpy.array_equal(b, b_expected)

def test_wrap_phase_arg():
    from test_utils.TestData import wrap_phase_arg
    phases = numpy.array([0, numpy.pi, -numpy.pi, 2*numpy.pi, -2*numpy.pi, numpy.pi/2, -numpy.pi/2])
    wrapped = wrap_phase_arg(phases)
    print("Original:", phases)
    print("Wrapped:", wrapped)