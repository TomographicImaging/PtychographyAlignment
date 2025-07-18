from ccpi.viewer import viewer2D
from ccpi.viewer.utils.conversion  import Converter 

class OpenViewer():
    def __init__(self, data):
        """Imports the data in a vtk viewer."""
        v = viewer2D()
        data_vtk = Converter.numpy2vtkImage(data)
        v.setInputData(data_vtk)
        v.startRenderLoop()