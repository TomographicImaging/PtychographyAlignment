from ccpi.viewer import viewer2D
from ccpi.viewer.utils.conversion  import Converter 

class OpenViewer():
    def __init__(self, data):
        v = viewer2D()
        # Create an image object
        data_vtk = Converter.numpy2vtkImage(data)
        v.setInputData(data_vtk)
        v.startRenderLoop()

def test_open_viewer():
    from io.Imports import ImportData
    ptytomofile = 'C:/Users/zvm34551/Coding_environment/DATA/Ptychography/pty_tomo_NX.h5'
    data = ImportData(ptytomofile)
    projections_raw= data.get_projections_raw()
    OpenViewer(projections_raw)

#test_open_viewer()
