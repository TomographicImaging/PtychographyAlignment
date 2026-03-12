import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

def plot_3axes(array, array_name='', data_order=None):
    plt.figure(figsize=(10,3))
    plt.subplot(131),plt.imshow(array[array.shape[0]//2,:,:]), plt.title(array_name), plt.colorbar()
    if data_order is not None:
        plt.xlabel(data_order[2])
        plt.ylabel(data_order[1])
    plt.subplot(132),plt.imshow(array[:,array.shape[1]//2,:]), plt.title(array_name), plt.colorbar()
    if data_order is not None:
        plt.xlabel(data_order[2])
        plt.ylabel(data_order[0])
    plt.subplot(133),plt.imshow(array[:,:,array.shape[2]//2]), plt.title(array_name), plt.colorbar()
    if data_order is not None:
        plt.xlabel(data_order[1])
        plt.ylabel(data_order[0])
    plt.tight_layout()
    plt.show()

def plot_crop(volume, Nx_start, Nx_stop, Ny_start, Ny_stop, Nangles):
    fig, ax = plt.subplots(figsize=(8,8))
    img = ax.imshow(volume[:,:,0])
    plt.colorbar(img, ax=ax)
    ax.set_xlabel('Nx')
    ax.set_ylabel('Ny')
    ax.plot([Nx_start, Nx_stop], [Ny_start, Ny_start], '--r')
    ax.plot([Nx_start, Nx_stop], [Ny_stop, Ny_stop], '--r')

    ax.plot([Nx_start, Nx_start], [Ny_start, Ny_stop], '--r')
    ax.plot([Nx_stop, Nx_stop], [Ny_start, Ny_stop], '--r')

    plt.close(fig)

    def update(slice_idx):
        img.set_data(volume[:,:,slice_idx])
        ax.set_title(f'Slice {slice_idx}')
        fig.canvas.draw_idle()
        display(fig)

    widgets.interact(update, slice_idx=(0, Nangles - 1, 1))