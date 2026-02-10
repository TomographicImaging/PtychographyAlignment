# %%
import os
import numpy as np 
import matplotlib.pyplot as plt

from gvxrPython3 import gvxr

from cil.recon import FBP
from cil.plugins.astra.processors import FBP as astra_FBP



# %%
generate_noise = True

# %% Create the experiment geometry
# Set up the source
print("Create an OpenGL context")
gvxr.createOpenGLContext()
print("Set up the beam")
gvxr.setSourcePosition(0.0,  -1.0, 0.0, "mm")
energy = 150
energy_units = "keV"
photons = 16000
gvxr.setMonoChromatic(energy, energy_units, photons)
gvxr.useParallelBeam()

# Set up the detector
print("Set up the detector")
gvxr.setDetectorPosition(0.0, 1.0, 0.0, "mm")
gvxr.setDetectorUpVector(0, 0, 1)
gvxr.setDetectorNumberOfPixels(300, 300)
gvxr.setDetectorPixelSize(1, 1, "um")

# %% Get a sample
simulation_name = "sphere_phantom"
gvxr.removePolygonMeshesFromSceneGraph()
large_sphere_radius = 100
gvxr.makeSphere(simulation_name, 100, 100, large_sphere_radius, "um")

x = [-60, 30, 20]
y = [-40, -30, 60]
z = [0, -5, 0]
sphere_radii = [20, 40, 30]
for i in np.arange(len(x)):
    sphere_name = f"sphere_{i}"
    gvxr.makeSphere(sphere_name, 50, 50, sphere_radii[i], "um")
    gvxr.translateNode(sphere_name, x[i], z[i], y[i], "um")
    gvxr.applyCurrentLocalTransformation(sphere_name)
    
    gvxr.addMesh(simulation_name, sphere_name)

gvxr.addPolygonMeshAsInnerSurface(simulation_name)
gvxr.setCompound(simulation_name, "SiO2")
gvxr.setDensity(simulation_name, 2.2,"g.cm-3")

# Compute an X-ray image
print("Compute an X-ray image")
gvxr.displayScene()
x_ray_image = np.array(gvxr.computeXRayImage()) / gvxr.getTotalEnergyWithDetectorResponse()
plt.imshow(x_ray_image), plt.xlabel('x'), plt.ylabel('y'), plt.colorbar()
# %% Generate the noise
start = 0
stop = 180
step = 0.5
include_last_angle = False

jitter_y = False

angle_set = np.linspace(start, stop, num=int((stop-start) / step), endpoint=include_last_angle)
xray_image_set = np.zeros((len(angle_set), gvxr.getDetectorNumberOfPixels()[1], gvxr.getDetectorNumberOfPixels()[0]))
delta_x = np.zeros(len(angle_set))
delta_y = np.zeros(len(angle_set))
if generate_noise:

    # random jitter
    rng = np.random.default_rng()
    max_jitter_x = 0.005*gvxr.getDetectorSize("um")[0] # 0.5% of sample size movement projection to projection
    max_jitter_y = 0.005*gvxr.getDetectorSize("um")[1] # 0.5% of sample size movement projection to projection
    damping = 0.02  # how strongly the system corrects random walk
    random_walk_x = 0.0
    random_walk_y = 0.0
    
    # thermal expansion
    delta_T = 2 # change in temperature
    alpha = 10 # coefficient of linear thermal expansion (~Al)
    expansion = 1*alpha*(delta_T) # total length change
    t_expansion = 300 # period over which the thermal expansion occurs (projections)
    delta_L = (1-expansion)/t_expansion # length change per projection
    thermal_x = 0.0
    thermal_y = 0.0

    # oscillation
    oscillation_frequency = 0.9
    oscillation_amplitude_x = 0.005*gvxr.getDetectorSize("um")[0] # 0.5% of sample size movement projection to projection
    oscillation_amplitude_y = 0.005*gvxr.getDetectorSize("um")[1] # 0.5% of sample size movement projection to projection
    osc_x = 0
    osc_y = 0
    
    for i in np.arange(len(angle_set)):
        jitter_step_x = rng.uniform(-max_jitter_x , max_jitter_x)
        jitter_step_y = rng.uniform(-max_jitter_y , max_jitter_y)
        # Damped random walk
        random_walk_x += jitter_step_x - damping * random_walk_x
        random_walk_y += jitter_step_y - damping * random_walk_y

        if i < t_expansion:
            thermal_x += delta_L
            thermal_y += delta_L

        osc_x += oscillation_amplitude_x * (0.7 * np.sin(oscillation_frequency * i) + 0.3 * np.cos(0.5*oscillation_frequency * i))
        osc_y += oscillation_amplitude_y * (0.7 * np.sin(oscillation_frequency * i) + 0.3 * np.cos(0.5*oscillation_frequency * i))

        osc_x += oscillation_amplitude_x * np.sin(oscillation_frequency * i)
        osc_y += oscillation_amplitude_y * np.sin(oscillation_frequency * i)

        delta_x[i] = int(thermal_x + random_walk_x + osc_x) # int for now
        delta_y[i] = int(thermal_y + random_walk_y + osc_y) # int for now
    
    if jitter_y == False:
        delta_y = np.zeros(len(angle_set))
    
    jitter_real_x = np.load('delta_x_1D_fullNth_1it.npy')
    jitter_real_y = np.load('delta_y_1D_fullNth_1it.npy')
    plt.plot(delta_x)
    plt.plot(delta_y)
    # plt.plot(jitter_real_x)
    plt.plot(jitter_real_y)

# %% Simulate a CT scan
for i in np.arange(len(angle_set)):
    # Rotate
    gvxr.rotateNode(simulation_name, angle_set[i], 0, 0, 1)

    # Shift
    print(delta_x[i], delta_y[i])
    # gvxr.translateNode(simulation_name, delta_x[i], 0, delta_y[i], "um")
    gvxr.setDetectorPosition(delta_x[i], gvxr.getDetectorPosition("um")[1], delta_y[i], "um")
    # Compute xray image
    xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single)/ gvxr.getTotalEnergyWithDetectorResponse()
    xray_image_set[i] = xray_image
    
    # print(gvxr.rota)
    # Restore the initial state
    # gvxr.translateNode(simulation_name, -delta_x[i], 0, -delta_y[i], "um")
    # gvxr.setDetectorPosition(-delta_x[i], gvxr.getDetectorPosition("um")[1], -delta_y[i], "um")
    gvxr.rotateNode(simulation_name, -angle_set[i], 0, 0, 1)

gvxr.setDetectorPosition(0, gvxr.getDetectorPosition("um")[1], 0, "um")
# islicer(xray_image_set)

plt.figure(figsize=(10,5))
plt.subplot(131),plt.imshow(xray_image_set[150,:,:]), plt.xlabel('x'), plt.ylabel('y')
plt.subplot(132),plt.imshow(xray_image_set[:,150,:]), plt.xlabel('x'), plt.ylabel('z')
plt.subplot(133),plt.imshow(xray_image_set[:,:,150]), plt.xlabel('y'), plt.ylabel('z')
# %%
# Apply Beer-Lambert law
np.log(xray_image_set, out=xray_image_set)
np.negative(xray_image_set,out=xray_image_set)
xray_image_set = xray_image_set.astype(np.float32)
# %%
plt.imshow(xray_image_set[150,:,:]), plt.xlabel('x'), plt.ylabel('y'), plt.colorbar()

# %% Check shifts are correct
def apply_x_shifts(projections, trans):
    for th in range(projections.shape[0]):
        shift = int(trans[th])
        projections[th,:,:] = np.roll(projections[th,:,:], shift, axis=1)
    return projections
def apply_y_shifts(projections, trans):
    for th in range(projections.shape[0]):
        shift = int(trans[th])
        projections[th,:,:] = np.roll(projections[th,:,:], shift, axis=0)
    return projections

xray_image_set_shifted = xray_image_set.copy()
# must be in-place
shifts_x = 0.5*delta_x
shifts_y = 0.5*delta_y
xray_image_set_shifted = apply_x_shifts(xray_image_set_shifted, shifts_x)
xray_image_set_shifted  = apply_y_shifts(xray_image_set_shifted , shifts_y)
# islicer(xray_image_set)

plt.figure(figsize=(10,5))
plt.subplot(131),plt.imshow(xray_image_set_shifted[150,:,:]), plt.xlabel('x'), plt.ylabel('y')
plt.subplot(132),plt.imshow(xray_image_set_shifted[:,150,:]), plt.xlabel('x'), plt.ylabel('z')
plt.subplot(133),plt.imshow(xray_image_set_shifted[:,:,150]), plt.xlabel('y'), plt.ylabel('z')

# %%
from cil.framework import AcquisitionGeometry, AcquisitionData
from cil.recon import FBP
from cil.utilities.display import show2D

ag = AcquisitionGeometry.create_Parallel3D().set_angles(angle_set).set_panel([300,300])
ag_shifted = AcquisitionGeometry.create_Parallel3D().set_angles(angle_set).set_panel([300,300])

data = AcquisitionData(xray_image_set, geometry=ag)
data_shifted = AcquisitionData(xray_image_set_shifted, geometry=ag_shifted)

data_shifted.reorder('astra')
data.reorder('astra')

fbp = FBP(data, backend='astra')
recon = fbp.run()
fbp = FBP(data_shifted, backend='astra')
recon_shifted = fbp.run()

show2D([recon, recon_shifted])
# %%
# xray_image_set.shape = [Nangles, Ny, Nx]

folder = '/mnt/share/ALC_ptychography_alignment/simulations/'
folder = 'output_data/'
np.save(folder+'sphere_phantom_360_projections_noy.npy', xray_image_set)
np.save(folder+'sphere_phantom_360_shifts_x_noy.npy', shifts_x)
np.save(folder+'sphere_phantom_360_shifts_y_noy.npy', shifts_y)
np.save(folder+'sphere_phantom_360_theta_noy.npy', angle_set)

# %%
np.save(folder+'sphere_phantom_360_projections_noy_shifted.npy', xray_image_set_shifted)
# %%
