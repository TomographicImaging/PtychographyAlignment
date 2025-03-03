import numpy as np
from OpenViewer import OpenViewer

class TestData():
    def __init__(self):
        self.data = self.create_phase()

    def wrap_phase_arg(self, phase):
        return np.angle(np.exp(1j * phase), deg=False)

    def create_phase(self):# Define dimensions
        (nx, ny, nz) = (100, 200, 300)  # (slices, height, width)
        # Create a 1D array from -2π to 2π
        phase_1d = np.linspace(-2 * np.pi, 2 * np.pi, nz)
        # Repeat the 1D array across the 3D volume (nx x ny x nz)
        phase_stack = np.tile(phase_1d, (nx, ny, 1))
        
        np.set_printoptions(suppress=True)
        #print(phase_stack)
        phase_stack = self.wrap_phase_arg(phase_stack)
        #print(phase_stack)
        return phase_stack
    
    def plot(self):# Create a plot
        wrapped_phase = TestData().data
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # Plot the wrapped phase values
        c = ax.imshow(wrapped_phase[:, :,50]/np.pi, cmap='gray', interpolation='nearest')  # Using the middle slice for visualization

        # Add a color bar
        fig.colorbar(c, ax=ax, label="Phase (radians)")

        # Add labels and title
        ax.set_title("Wrapped Phase Plot")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.show()



#wrapped_phase = TestData().data
    