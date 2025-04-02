# %%
import os
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import json
font = {'family' : 'serif',
         'size'   : 15
       }
matplotlib.rc('font', **font)
from tifffile import imwrite
from gvxrPython3 import gvxr
from gvxrPython3 import gvxr2json
from gvxrPython3.JSON2gVXRDataReader import *
from cil.recon import FBP
from cil.plugins.astra.processors import FBP as astra_FBP
from cil.processors import TransmissionAbsorptionConverter
from cil.utilities.display import show_geometry, show2D
from cil.utilities.jupyter import islicer
from scipy.spatial.transform import Rotation as R

from cil.io import TIFFWriter

# %%
simulation_name = "Turtle"
output_path = "output_data"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# %% Create the experiment geometry
# Set up the source
print("Create an OpenGL context")
gvxr.createOpenGLContext()
print("Set up the beam")
gvxr.setSourcePosition(0.0,  -40.0, 0.0, "mm")
gvxr.setMonoChromatic(500, "keV", 1000)
gvxr.useParallelBeam()

# Set up the detector
print("Set up the detector")
gvxr.setDetectorPosition(0.0, 40.0, 0.0, "mm")
gvxr.setDetectorUpVector(0, 0, 1)
gvxr.setDetectorNumberOfPixels(300, 300)
gvxr.setDetectorPixelSize(0.5, 0.5, "mm")

# %% Get a sample
# Locate the sample STL file
fname =  "../CCPi-Diamond-Laminography/Turtle_Singlecolor.stl"

# Load the sample data
if not os.path.exists(fname):
    raise IOError(fname)

print("Load the mesh data from", fname)
gvxr.loadMeshFile(simulation_name, fname, "mm")

print("Move ",simulation_name, " to the centre")
gvxr.moveToCentre(simulation_name)
gvxr.applyCurrentLocalTransformation(simulation_name)

# Choose a density for the sample
# Carbon (Z number: 6, symbol: C)
gvxr.setElement(simulation_name, 6)
gvxr.setElement(simulation_name, "C")

# Compute an X-ray image
print("Compute an X-ray image")
gvxr.displayScene()
x_ray_image = np.array(gvxr.computeXRayImage()) / gvxr.getTotalEnergyWithDetectorResponse()
show2D(x_ray_image)
# %% Simulate a CT scan
start = 0
stop = 360
step = 1
angle_set = np.arange(start, stop, step)
xray_image_set = np.zeros((stop, gvxr.getDetectorNumberOfPixels()[1], gvxr.getDetectorNumberOfPixels()[0]))
rng = np.random.default_rng()
max_jitter = 20
for i in angle_set:
    # Rotate
    gvxr.rotateNode(simulation_name, step, 0, 0, 1)

    # Shift
    shift_x = (rng.random()*(max_jitter*2))-max_jitter
    shift_y = (rng.random()*(max_jitter*2))-max_jitter
    print(shift_x, shift_y)
    gvxr.translateNode(simulation_name, shift_x, 0, shift_y, "mm")

    # Compute xray image
    xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
    xray_image_set[i] = xray_image

    # Restore the initial state
    gvxr.translateNode(simulation_name, -shift_x, 0, -shift_y, "mm")

islicer(xray_image_set)
# %% Save the simulated projections as tiff files
prefix = simulation_name + "_jitter_simulation_"
sub_folder = prefix + str(len(xray_image_set))

if not os.path.exists(os.path.join(output_path, sub_folder)):
    os.makedirs(os.path.join(output_path, sub_folder))

for i, img in enumerate(xray_image_set):
    fname = os.path.join(output_path, sub_folder, prefix + str(i).zfill(4) + ".tif")
    imwrite(fname, img.astype(np.float32), photometric='minisblack')

# %% Save the current simulation states in a JSON file.
json_fname = os.path.join(output_path, prefix + str(len(xray_image_set)) + ".json")
gvxr2json.saveJSON(json_fname)
with open(json_fname) as f:
    params = json.load(f)
params["Scan"] = {
    "OutFolder": sub_folder,
    "NumberOfProjections": len(xray_image_set),
    "AngleStep": step,
    "StartAngle": start,
    "FinalAngle": stop,
    "IncludeLastAngle": True,  
    "Flat-Field Correction": False,
    "CentreOfRotation": list(gvxr.getCentreOfRotationPositionCT("mm")) + ["mm"],
    "RotationAxis": list(gvxr.getDetectorUpVector())
}
with open(json_fname, "w") as file:
    json.dump(params, file, indent=4)

# %% Read the simulated data with CIL
from gvxrPython3.JSON2gVXRDataReader import *
reader = JSON2gVXRDataReader(json_fname)
data = reader.read()

# Apply Beer-Lambert law
data = TransmissionAbsorptionConverter()(data)

# Check the data and geometry look right in CIL
data.reorder('tigre')
show_geometry(data.geometry)
islicer(data)

# %% Compare CIL recon FBP with astra
from cil.recon import FBP
data.reorder('astra')
fbp = FBP(data, backend='astra')
recon = fbp.run()
recon.apply_circular_mask(0.9)
show2D(recon)

# %% Scroll through the reconstruction
islicer(recon)