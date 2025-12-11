# %%
import os
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import json
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
simulation_name = "Spheres"
output_path = "output_data"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# %% Create the experiment geometry
# Set up the source
print("Create an OpenGL context")
gvxr.createOpenGLContext()
print("Set up the beam")
gvxr.setSourcePosition(0.0,  -5.0, 0.0, "mm")
energy = 150
energy_units = "keV"
photons = 16000
gvxr.setMonoChromatic(energy, energy_units, photons)
gvxr.useParallelBeam()

# Set up the detector
print("Set up the detector")
gvxr.setDetectorPosition(0.0, 5.0, 0.0, "mm")
gvxr.setDetectorUpVector(0, 0, 1)
gvxr.setDetectorNumberOfPixels(300, 300)
gvxr.setDetectorPixelSize(0.5, 0.5, "um")

# %% Get a sample
gvxr.makeSphere(simulation_name, 10, 10, 10, "um")

for i in [-3, -2, -1, 1, 2, 3]:
    sphere_name = f"sphere_{i}"
    gvxr.makeSphere(sphere_name, 10, 10, 10, "um")
    gvxr.translateNode(sphere_name, i*20, 0, 0, "um")
    gvxr.applyCurrentLocalTransformation(sphere_name)
    
    gvxr.addMesh(simulation_name, sphere_name)

gvxr.addPolygonMeshAsInnerSurface(simulation_name)
gvxr.setCompound(simulation_name, "SiO2")
gvxr.setDensity(simulation_name, 2.2,"g.cm-3")

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
max_jitter_x = 0#3
max_jitter_y = 0#1
jitter_x = np.zeros(len(angle_set))
jitter_y = np.zeros(len(angle_set))
for i in angle_set:
    # Rotate
    gvxr.rotateNode(simulation_name, step, 0, 0, 1)

    # Shift
    jitter_x[i] = (rng.random()*(max_jitter_x*2))-max_jitter_x
    jitter_y[i] = (rng.random()*(max_jitter_y*2))-max_jitter_y
    print(jitter_x[i], jitter_y[i])
    gvxr.translateNode(simulation_name, jitter_x[i], 0, jitter_y[i], "um")

    # Compute xray image
    xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
    xray_image_set[i] = xray_image

    # Restore the initial state
    gvxr.translateNode(simulation_name, -jitter_x[i], 0, -jitter_y[i], "um")

islicer(xray_image_set)
# %% Save the simulated projections as tiff files
# prefix = 
sub_folder = simulation_name + "_simulation_" + str(len(xray_image_set))

if not os.path.exists(os.path.join(output_path, sub_folder)):
    os.makedirs(os.path.join(output_path, sub_folder))

for i, img in enumerate(xray_image_set):
    fname = os.path.join(output_path, sub_folder, simulation_name + "_simulation_" + str(i).zfill(4) + ".tif")
    imwrite(fname, img.astype(np.float32), photometric='minisblack')

# save the jitter arrays
np.save(simulation_name + "_delta_x.npy", jitter_x)
np.save(simulation_name + "_delta_y.npy", jitter_y)

# %% Save the current simulation states in a JSON file.
json_fname = os.path.join(output_path, simulation_name + "_simulation_" + str(len(xray_image_set)) + ".json") 
# gvxr2json.saveJSON(json_fname) # This doesn't work when there is no STL file, just make the json file manually
# with open(json_fname) as f:
#     params = json.load(f)
params = {}
params["Window size"] = list(gvxr.getWindowSize()),
params["Source"] = {
    "Position": list(gvxr.getSourcePosition("mm")) + ["mm"],
    "Shape" : "PARALLEL",
    "Beam":list({
        "Energy": energy,
        "PhotonCount": photons,
        "Unit": energy_units
    })
}
params["Detector"] = {
    "Position" : list(gvxr.getDetectorPosition("mm")) + ["mm"],
    "UpVector" : list(gvxr.getDetectorUpVector()),
    "RightVector" : list(gvxr.getDetectorRightVector()),
    "NumberOfPixels" : list(gvxr.getDetectorNumberOfPixels()),
    "Size" : list(gvxr.getDetectorSize("mm")) + ["mm"]

}
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
print(params)
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
recon.apply_circular_mask(1)
show2D(recon)

# %% Scroll through the reconstruction
islicer(recon)