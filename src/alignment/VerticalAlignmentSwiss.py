import numpy as np
import matplotlib.pyplot as plt

# Load the saved array
projections_reduced = np.load(r"src\projections_reduced.npy")
print("angle, vertical, horizontal = ",projections_reduced.shape)
#If swapping x and y
swap_xy =True #False
if swap_xy ==True:
    projections_reduced = np.swapaxes(projections_reduced, 1, 2)
    align = "X" 
    other = "Y"
else:
    align = "Y"
    other = "X"

angles_reduced = np.load(r"src\angles_reduced.npy")

def plot_array(arr,title = "Plot",x_label=other,y_label=align):# Create a plot
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

def plot_1D(arr1D,title = "Plot",x_label=other,y_label=align, label = 'average'):# Create a plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    c = ax.plot(arr1D, label=label, color='blue')  # Using the middle slice for visualization
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.show()

class VerticalAlignmentSwiss():
    def __init__(self):
        #self.data = self.create_data()
        self.data = projections_reduced
        plot_array(self.data[0], "0th projection", other,align)
        self.ROI = self.select_ROI(self.data)
        plot_array(self.ROI[0], "0th projection ROI", other,align)
        M = self.calculate_M()
        psi = self.remove_legendre_terms_matrix(M, degree=1)
        self.test_shift_psi_by_array(psi)
        self.test_compute_error(psi)
        self.delta_y_1D_final = self.align_projections(psi)
        psi_final = self.shift_psi_by_array(psi, self.delta_y_1D_final)[0]
        plot_array(np.transpose(psi_final), f"psi_theta({align}) shifted",  'projection #', align)

    def create_data(self):# Define dimensions
        (nx, ny, nth) = (2, 3, 4)  # (slices, height, width)
        # Create a 1D array from -2π to 2π
        #phase_1d = np.linspace(-1* np.pi, 1 * np.pi, nth)
        phase_1d = np.linspace(0, 3, nth)
        # Repeat the 1D array across the 3D volume (nx x ny x nz)
        phase_stack = np.tile(phase_1d, (nx, ny, 1))
        
        #np.set_printoptions(suppress=True)
        return phase_stack
    
    def select_ROI(self, arr):
        if swap_xy == True:
            return arr[:,:,:]
        else:
            return arr[:,100:600,:]

    def calculate_M(self):
        arr = self.ROI
        #print(arr[0])
        M = np.sum(arr, axis=2)
        plot_array(np.transpose(M), f"M_theta({align})",  'projection #', align)
        return M
    
    def remove_legendre_terms(f_y, degree=1):
        """
        Remove Legendre polynomial terms from f_y using least-squares fitting.
        np.polynomial.legendre.legfit does perform a least-squares fit when
        determining the coefficients of the Legendre polynomial.

        Parameters:
        y      : 1D array (y-coordinates)
        f_y    : 1D array (Mass distribution function Mθ(y))
        degree : int (Degree of Legendre polynomial to remove)

        Returns:
        psi : 1D array (f_y with Legendre polynomial components removed)
        """

        # Normalize y to [-1, 1] for Legendre polynomials
        y = np.arange(len(f_y))
        y_norm = 2 * (y - y.min()) / (y.max() - y.min()) - 1  # normalize to [-1, 1]
        y_norm = 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1  

        # Fit Legendre polynomial of given degree
        coeffs = np.polynomial.legendre.legfit(y_norm, f_y, degree)

        # Evaluate fitted polynomial
        f_legendre = np.polynomial.legendre.legval(y_norm, coeffs)

        # Subtract the fitted Legendre polynomial from M_y
        psi = f_y - f_legendre

        return psi
    
    def remove_legendre_terms_matrix(self,M_y, degree=1):
        from numpy.polynomial.legendre import legvander, legval
        """
        Vectorized removal of Legendre polynomial trends from each row of M_y.

        Parameters:
        M_y    : 2D array (shape: N x H), each row is a signal over y
        degree : int, degree of Legendre polynomial to remove

        Returns:
        psi    : 2D array of same shape with Legendre trends removed
        """
        Ny = M_y.shape[1]
        y = np.arange(Ny)
        y_norm = 2 * (y - y.min()) / (y.max() - y.min()) - 1  # normalize to [-1, 1]

        # Construct Legendre Vandermonde matrix (H x degree+1)
        V = legvander(y_norm, degree)  # shape: (H, degree+1)


        # Least squares fit for all rows at once
        coeffs = np.linalg.lstsq(V, M_y.T, rcond=None)[0]  # shape: (degree+1, N)

        # Reconstruct fitted trends
        f_legendre = V @ coeffs  # shape: (H, N)

        # Subtract trends and return result
        psi = M_y.T - f_legendre  # shape: (H, N)
        plot_array(np.transpose(psi.T), f"psi_theta({align})",  'projection #', align)
        return psi.T  # shape back to (N, H)
    
    def shift_psi_by_array(self, psi, shifts_1D):
        "shift psi by 1d array of y shifts and calculate mean for this, over angle"
        Nth, Ny = psi.shape  # Get dimensions
        # Compute the average projection over all θ, using the current shifts
        shifted_psi = np.array([np.roll(psi[theta, :], -shifts_1D[theta]) for theta in range(Nth)])
        avg_psi = np.mean(shifted_psi, axis=0)  # Compute mean along θ
        return shifted_psi, avg_psi
        
    def compute_error(self, psi, shifts_1D, theta_index, new_shift):
        """
        Compute E² error for a given projection (theta_idx) using a test shift (new_shift),
        comparing against the average of all projections shifted by shifts_old.

        Parameters:
        psi         : (Nth, Ny) array of projections
        theta_idx   : int, index of projection being optimized
        new_shift   : int, test shift value for this projection
        shifts_old  : (Nth,) array of current shifts from previous iteration

        Returns:
        error       : scalar, total squared difference over y and θ
        """
        # Step 1: Build new shift array: same as old, except for theta_idx
        shifts_new_1D = shifts_1D.copy()
        shifts_new_1D[theta_index] = new_shift

        avg_psi = self.shift_psi_by_array(psi, shifts_1D)[1]

        # Apply new test shift to the current projection
        new_psi_matrix = self.shift_psi_by_array(psi, shifts_new_1D)[0]
        #print(new_psi_matrix)
        

        # Compute squared error
        #print(new_psi_matrix - avg_psi)
        #a = np.array([[1,1,1],[1,1,1]]) - np.array([1,1,1])
        #print(a)

        error = np.sum((new_psi_matrix - avg_psi) ** 2)
        
        return error
    
    def compute_error_vectorised(self, psi, shifts_1D, theta_index, new_shift_array):
        "WIP"
        avg_psi = self.shift_psi_by_array(psi, shifts_1D)[1]

        # Step 1: Build new shift array: same as old, except for theta_idx
        shifts_new_1D = shifts_1D.copy()
        shifts_new_1D[theta_index] = new_shift
        # Apply new test shift to the current projection
        new_psi_matrix = self.shift_psi_by_array(psi, shifts_new_1D)[0]
        error = np.sum((new_psi_matrix - avg_psi) ** 2)
        #print(new_psi_matrix)
    
    def align_projections(self, psi, max_shift=25, iterations=1):
        """
        Iteratively align projections by minimizing E^2.

        Parameters:
        psi        : 2D NumPy array (Y x Θ), projection data
        max_shift  : int, max pixel shift to search in each direction, only for the first iteration. In the successive iterations this value is updated.
        iterations : int, number of iterations

        Returns:
        delta_y : 1D NumPy array (Θ,), optimized shifts per projection
        """
        Nth, Ny = psi.shape
        print("psi shape is ",psi.shape)
        delta_y_1D = np.zeros(Nth, dtype=int)  # Initialize shifts to zero

        for _ in range(iterations):  # Iterate to refine alignment
            for theta in range(1, 3):#Nth):  # Process each projection separately, 0 is the reference
                #print("theta iteration is ", theta)
                shift_range = range(-max_shift, max_shift + 1)
                #print("shift range is", shift_range)
                errors_1D = [self.compute_error(psi, delta_y_1D, theta, scalar_shift) for scalar_shift in shift_range]
                #print("errors1D is",errors_1D)
                best_shift = shift_range[np.argmin(errors_1D)]  # Find shift that minimizes error
                #print("best shift for theta", theta, "is",best_shift)
                delta_y_1D[theta] = best_shift  # Update shift for this projection
            print(delta_y_1D)
            max_shift = 2*np.max(delta_y_1D)

        return delta_y_1D

    def test_shift_psi_by_array(self, psi):
        Nth, Ny = psi.shape
        delta_y_1D = np.zeros(Nth, dtype=int)  # Initialize shifts to zero
        for theta in range(Nth//2):  # Process each projection separately
            delta_y_1D[theta] = 200
        # Compute the average projection over all θ, using the current shifts
        shifted_projections, avg = self.shift_psi_by_array(psi, delta_y_1D)
        #print("avg is ",avg)
        plot_1D(avg,"Average over the projection angle theta",align, "Average")
        plot_array(np.transpose(shifted_projections), f"psi_theta({align}) shifted",  'projection #', align)

    def test_compute_error(self, psi):
        Nth, Ny = psi.shape
        delta_y_1D = np.zeros(Nth, dtype=int)  # Initialize shifts to zero
        error = self.compute_error(psi, delta_y_1D, 0, 1) 
        print("error is ", error)
                

def plot():# Create a plot
    data = VerticalAlignmentSwiss().data
    #print(data.shape)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # Plot the wrapped phase values
    #c = ax.imshow(data[0, :,:]/np.pi, cmap='gray', interpolation='nearest')  # Using the middle slice for visualization
    c = ax.imshow(data[0, :,:], cmap='gray', interpolation='nearest')  # Using the middle slice for visualization

    # Add a color bar
    fig.colorbar(c, ax=ax, label="Phase (radians)")

    # Add labels and title
    ax.set_title("Phase Plot")
    ax.set_xlabel(other)
    ax.set_ylabel(align)

    plt.show()




def test_alignment():
    va= VerticalAlignmentSwiss()
    if swap_xy ==True:
        np.save(r"src\delta_x_1D.npy", va.delta_y_1D_final)
    else:
        np.save(r"src\delta_y_1D.npy", va.delta_y_1D_final)
    
test_alignment()






