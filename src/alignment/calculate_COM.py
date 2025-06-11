
# %%
import numpy as np
import os

cwd = os.getcwd()
print("Current working dir:", cwd)
# %%

#pp = np.load(r"C:\Users\zvm34551\Coding_environment\PtychographyAlignment\data\experimental\projections_aligned_horiz_and_vert.npy")
#pp = np.load(r"C:\Users\zvm34551\Coding_environment\PtychographyAlignment\data\experimental\data\projections_reduced.npy")
#print(pp.shape)
#pp = pp[1:,700:1430,1300:2300]
#pp = pp[1:,700:1430,1300:2300]
pp_noj = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250416\sphere_phantom_nojitter_simulation_361_projections.npy")
pp_j = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250416\sphere_phantom_jitter_simulation_361_projections.npy")
# %%
pp_jy = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250416\sphere_phantom_yjitter_simulation_361_projections.npy")#
pp_jx = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250416\sphere_phantom_xjitter_simulation_361_projections.npy")


import matplotlib.pyplot as plt
def plot_array(arr,title = "Plot",x_label='x',y_label='angle'):# Create a plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    c = ax.imshow(arr, cmap='gray', interpolation='nearest')  # Using the middle slice for visualization
    # Add a color bar
    fig.colorbar(c, ax=ax, label="Phase (radians)")
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.show()
# %%
#plot_array(pp[:,900,:])

def calc_R(pp):
    """from paper Tomographic image alignment in three-dimensional coherent diffraction microscopy eq (1)"""
    Nth, Ny, Nx = pp.shape
    x_array = np.arange(Nx) #x=i
    y_array = np.arange(Ny) #y=j
    #print(x_array)
    # %%
    # sum over y-axis to collapse each projection into a 1D profile along x
    denominator = pp.sum(axis=1)
    print(denominator.shape)  # shape (Nth, Nx)
    #m0_total = m0.sum(axis=1)
    #print(m0_total)
    #print(m0_total.shape)
    # %%
    tc = (m0 * x_array).sum(axis=2) / denominator  # shape (Nth,)
    #print(tc)
    return tc

# %%
def calc_tc(pp):
    """COM in x"""
    Nth, Ny, Nx = pp.shape
    x_array = np.arange(Nx)
    #print(x_array)
    # %%
    # sum over y-axis to collapse each projection into a 1D profile along x
    m0 = pp.sum(axis=1)
    #print(m0)  # shape (Nth, Nx)
    m0_total = m0.sum(axis=1)
    #print(m0_total)
    #print(m0_total.shape)
    # %%
    tc = (m0 * x_array).sum(axis=1) / m0_total  # shape (Nth,)
    #print(tc)
    return tc

def calc_tax2(pp):
    """COM in y"""
    Nth, Ny, Nx = pp.shape
    y_array = np.arange(Ny)
    #print(x_array)
    # %%
    # sum over y-axis to collapse each projection into a 1D profile along x
    m0 = pp.sum(axis=2)
    #print(m0)  # shape (Nth, Nx)
    m0_total = m0.sum(axis=1)
    #print(m0_total)
    #print(m0_total.shape)
    # %%
    tc = (m0 * y_array).sum(axis=1) / m0_total  # shape (Nth,)
    #print(tc)
    return tc

# %%
def plot_1D(arr1D,title = "Plot com",x_label='angle',y_label='x', label = 'label'):# Create a plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    c = ax.plot(arr1D, label=label, color='blue')  # Using the middle slice for visualization
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.show()

tc_noj = calc_tc(pp_noj)
tc_jx = calc_tc(pp_jx)
tc_jx = calc_tc(pp_jx)
tc_jy = calc_tc(pp_jy)
tc_j = calc_tc(pp_j)
#plot_1D(tc_noj)
#plot_1D(tc_jx)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
c = ax.plot(tc_noj, color='blue')  # Using the middle slice for visualization
c = ax.plot(tc_jx, color='red')  # Using the middle slice for visualization
c = ax.plot(tc_j, color='green')  # Using the middle slice for visualization
c = ax.plot(tc_jy, color='pink')  # Using the middle slice for visualization
plt.show()

tc_noj = calc_tax2(pp_noj)
tc_jx = calc_tax2(pp_jx)
tc_jx = calc_tax2(pp_jx)
tc_jy = calc_tax2(pp_jy)
tc_j = calc_tax2(pp_j)
#plot_1D(tc_noj)
#plot_1D(tc_jx)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
c = ax.plot(tc_noj, color='blue')  # Using the middle slice for visualization
c = ax.plot(tc_jx, color='red')  # Using the middle slice for visualization
c = ax.plot(tc_j, color='green')  # Using the middle slice for visualization
c = ax.plot(tc_jy, color='pink')  # Using the middle slice for visualization
# Add labels and title
# Add labels and title
plt.show()

import numpy as np
from scipy.fft import fftn, ifftn, fftshift, ifftshift

imgtest_y = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])#, [3,3,3]])
print(imgtest_y)
imgtest_x = np.array(np.transpose([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))#, [3,3,3]])
print(imgtest_x)
def fourier_shift(img, dy, dx):
    """ This code runs sub-pixel circular shift based on Fourier factoring
    https://www.physics.ucla.edu/research/imaging/ProjectionAlignment/
    When the shifts dx and dy are integers, this code should circularly shift the image by exactly
    that number of pixels along each axis — with no interpolation or blurring.
    It shifts the 2D arrays left and right, up and down by the amount dy and dx"""
    ny, nx = img.shape

    x = np.arange(-((nx - 1) // 2), ((nx) // 2) + 1)
    y = np.arange(-((ny - 1) // 2), ((ny) // 2) + 1)
    print(x,y)
    X, Y = np.meshgrid(x, y)
    print(X)
    print(Y)

    # Compute the Fourier transform of the image with proper centering
    F = fftshift(ifftn(ifftshift(img)))

    # Construct the Fourier shift factor (phase ramp)
    Pfactor = np.exp(2j * np.pi * (dx * X / nx + dy * Y / ny))

    # Apply the shift in Fourier domain and invert the transform
    img2 = fftshift(fftn(ifftshift(F * Pfactor)))

    # Return the real part (image should be real-valued)
    return np.real(img2)

imgtest_y2= fourier_shift(imgtest_y, 0.5, 0)
print("result",imgtest_y2)
imgtest_x2= fourier_shift(imgtest_x, 0, 0.5)
print("result",imgtest_x2)



