
# %%
import numpy as np
import os

cwd = os.getcwd()
print("Current working dir:", cwd)
# %%

pp = np.load(r"..\projections_aligned_horiz_and_vert.npy")
pp = pp[1:,700:1430,1300:2300]
#pp = np.load(r"..\projections_reduced.npy")

# %%
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
# %%
Nth, Ny, Nx = pp.shape
x_array = np.arange(Nx)
print(x_array)
# %%
# sum over y-axis to collapse each projection into a 1D profile along x
m0 = pp.sum(axis=1)
print(m0)  # shape (Nth, Nx)
m0_total = m0.sum(axis=1)
print(m0_total)
print(m0_total.shape)
# %%
tc = (m0 * x_array).sum(axis=1) / m0_total  # shape (Nth,)
print(tc)

# %%
def plot_1D(arr1D,title = "Plot",x_label='x',y_label='y', label = 'label'):# Create a plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    c = ax.plot(arr1D, label=label, color='blue')  # Using the middle slice for visualization
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.show()

plot_1D(tc)
# %%
