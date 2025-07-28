# author: Danica Sugic
# This code was generated from the paper "Phase tomography from x-ray coherent diffractive imaging projections"
# DOI: 10.1364/OE.19.021345 (https://opg.optica.org/oe/fulltext.cfm?uri=oe-19-22-21345&id=223191) 
# The methods are work in progress. They were partially tested on the pollen-Volpe data and the simulated gVXR data.
# The error minimiser method in particular needs improvement.
#-------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Load the saved array
#projections = np.load(r"src\projections_reduced.npy")
#print("angle, vertical, horizontal = ",projections.shape)
#swap_xy =True #False# True #
#angles_reduced = np.load(r"src\angles_reduced.npy")



class VerticalAlignmentSwiss():
    def __init__(self, projections, max_shift = 50, iterations = 1, swap_xy = False, plotting = False, saving = False, result_directory = "."):
        """
        Initialises the attributes. The alignment direction can be swapped by changing the flag "swap_xy".

        Parameters
        ----------
        projections : 3D array
            input 3D image to be aligned
        max_shift : int
            max pixel shift to search in each direction
        iterations : int
            number of iterations to refine the alignment procedure
        swap_xy : bool
            this flag controls the alignment direction
        plotting : bool
            this flag activates the plotting 
        saving : bool
             if True, the shift results will be written to `result_directory`
        result_directory : str
            Path to the folder where results will be saved. This can be an absolute
            or relative path. If saving is False, this has no effect.
        """       
        self.data = projections
        self.max_shift = max_shift
        self.iterations = iterations
        self.swap_xy = swap_xy
        self.plotting = plotting
        self.saving = saving
        if self.saving == True:
            import os
            if self.swap_xy == True:
                filename = "delta_x_1D.npy"
            else:
                filename = "delta_y_1D.npy"
            self.save_path = os.path.join(result_directory, filename)
            if os.path.exists(self.save_path):
                raise FileExistsError(f"{self.save_path} already exists. Please rename your file.")
        if swap_xy ==True:
            projections = np.swapaxes(projections, 1, 2)
            self.align = "X" 
            self.other = "Y"
        else:
            self.align = "Y"
            self.other = "X"
        if plotting == True:
            self.plot_array(self.data[0], "0th projection", self.other, self.align)
        self.ROI = self.select_ROI(self.data, swap_xy)
        if plotting == True:
            self.plot_array(self.ROI[0], "0th projection ROI", self.other, self.align)
        
    def run_alignment(self):
        """Runs the pipeline for the alignment procedure."""
        M = self.calculate_M()
        psi = self.remove_legendre_terms_matrix(M, degree=1)
        self.delta_y_1D_final = self.calculate_shift_array(psi, self.max_shift, self.iterations)
        psi_final = self.shift_psi_by_array_and_mean(psi, self.delta_y_1D_final)[0]
        if self.plotting == True:
            self.plot_array(np.transpose(psi_final), f"psi_theta({self.align}) shifted",  'projection #', self.align)
        if self.saving == True:
            np.save(self.save_path, self.delta_y_1D_final)

    def plot_array(self, arr, title, x_label, y_label):
        """Creates a greyscale density plot of a 2D image.
        
        Parameters
        ----------
        arr : 2D scalar data to plot
        title : title of the plot
        x_label : horizontal label
        y_label : vertical label
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        c = ax.imshow(arr, cmap='gray', interpolation='nearest')  
        fig.colorbar(c, ax=ax, label="Phase (radians)")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.show()

    def plot_1D(self, arr1D, title, x_label, y_label, label = 'mean'):
        """Creates a line plot of a 1D array.
        
        Parameters
        ----------
        arr1D : 1D scalar data to plot
        title : title of the plot
        x_label : horizontal label
        y_label : vertical label
        label : data label
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        c = ax.plot(arr1D, label=label, color='blue')  
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.show()

    def select_ROI(self, arr, swap_xy, roi=None):
        """
        Select a Region of Interest (ROI) from a 3D array.

        Parameters
        ----------
        arr : ndarray
            Input 3D array.
        swap_xy : bool
            If True, skip cropping.
        roi : tuple or None
            A tuple of slice ranges for cropping in the form (start, end),
            applied along the second axis (e.g., Y). For example, (100, 200).
            If None, no cropping is applied.

        Returns
        -------
        ndarray
            Cropped or original array.
        """
        if swap_xy == True:
            return arr[:,:,:]
        else:
            if roi is not None:
                start, end = roi
                return arr[:, start:end, :]
            else:
                return arr[:,:,:]
                #return arr[:,100:200,:]
                #return arr[:,100:600,:]

    def calculate_M(self):
        """
        Compute and plot the M projection by summing the ROI along the x-axis.

        This method calculates a 2D projection `M` from a 3D region of interest (ROI)
        by summing across the third axis (axis=2), effectively collapsing the x-dimension.
        The result is then transposed and passed to the `plot_array` method for visualisation.

        Returns
        -------
        np.ndarray
            A 2D array `M` of shape (Nθ, Ny), representing the integrated projection along x.
        """
        arr = self.ROI
        M = np.sum(arr, axis=2)
        if self.plotting == True:
            self.plot_array(np.transpose(M), f"M_theta({self.align})",  'projection #', self.align)
        return M
    
    def remove_legendre_terms(f_y, degree=1):
        """
        Removes Legendre polynomial components from a 1D signal using least-squares fitting.
        Legendre polynomials are defined over the interval [-1, 1], so the input coordinates 
        must be normalised to this range before fitting. This method internally generates
        normalised coordinates corresponding to the indices of `f_y`, fits a Legendre polynomial 
        of the specified degree to the signal using least-squares, and subtracts the fitted 
        polynomial to return the residual.

        Parameters
        ----------
        f_y : np.ndarray
            1D array representing the signal (e.g., mass distribution Mθ(y)).
        degree : int, optional
            Degree of the Legendre polynomial to remove. Default is 1 (linear trend).

        Returns
        -------
        np.ndarray
            Residual signal with Legendre polynomial components removed.
        """

        y = np.arange(len(f_y))
        y_norm = 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1  
        coeffs = np.polynomial.legendre.legfit(y_norm, f_y, degree)
        f_legendre = np.polynomial.legendre.legval(y_norm, coeffs)
        psi = f_y - f_legendre
        return psi
    
    def remove_legendre_terms_matrix(self, M_y, degree=1):
        """
        Removes Legendre polynomial trends from each row of a 2D signal matrix using 
        a vectorized least-squares approach.

        Each row of `M_y` is treated as a 1D signal over the y-axis. Since Legendre 
        polynomials are defined on the interval [-1, 1], the y-axis positions are 
        internally normalised to this range. A Legendre Vandermonde matrix `V`, with shape (H, degree+1),
        is constructed, and a least-squares fit is applied to all rows simultaneously to remove trends 
        up to the specified polynomial degree.

        Parameters
        ----------
        M_y : np.ndarray
            2D array of shape (N, H), where each row is a signal along the y-axis.
        degree : int, optional
            Degree of the Legendre polynomial to remove. Default is 1 (linear trend).

        Returns
        -------
        np.ndarray
            2D array of the same shape as `M_y`, with Legendre polynomial components removed
            from each row.
        """
        from numpy.polynomial.legendre import legvander, legval

        Ny = M_y.shape[1]
        y = np.arange(Ny)
        y_norm = 2 * (y - y.min()) / (y.max() - y.min()) - 1

        V = legvander(y_norm, degree)
        coeffs = np.linalg.lstsq(V, M_y.T, rcond=None)[0]  # shape: (degree+1, N)
        f_legendre = V @ coeffs  # shape: (H, N)
        psi = M_y.T - f_legendre  # shape: (H, N)
        if self.plotting == True:
            self.plot_array(np.transpose(psi.T), f"psi_theta({self.align})",  'projection #', self.align)
        return psi.T  # shape back to (N, H)
        
    def compute_error(self, psi, shifts_1D, theta_index, new_shift):
        """
        Applies the new shift to the old shift. Calculates the mean psi with
        the old shift. Calculates psi with the new shift applied.
        Computes E² error for a given projection (theta_index) using the new_shift,
        comparing against the mean of all projections shifted by the old shifts.

        Parameters:
        -----------
        psi : (Nth, Ny) array
            calculated from M by removing the Legendre polynomials
        shifts_1D : (Nth,) array of current shifts from previous iteration
        theta_index : int, index of projection being optimized
        new_shift : int, test shift value for this projection
        

        Returns:
        --------
        error : scalar, total squared difference over y and θ
        """
        shifts_new_1D = shifts_1D.copy()
        shifts_new_1D[theta_index] = new_shift
        mean_psi = self.shift_psi_by_array_and_mean(psi, shifts_1D)[1]
        new_psi_matrix = self.shift_psi_by_array_and_mean(psi, shifts_new_1D)[0]
        error = np.sum((new_psi_matrix - mean_psi) ** 2)
        return error
    
    def compute_error_vectorised(self, psi, shifts_1D, theta_index, new_shift_array):
        "25/07/2025 This method is WIP. It is not used in the pipeline."
        mean_psi = self.shift_psi_by_array_and_mean(psi, shifts_1D)[1]

        # Step 1: Build new shift array: same as old, except for theta_idx
        shifts_new_1D = shifts_1D.copy()
        shifts_new_1D[theta_index] = new_shift
        # Apply new test shift to the current projection
        new_psi_matrix = self.shift_psi_by_array_and_mean(psi, shifts_new_1D)[0]
        error = np.sum((new_psi_matrix - mean_psi) ** 2)
        #print(new_psi_matrix)
    
    def shift_psi_by_array_and_mean(self, psi, shifts_1D):
        """
        Shifts each row of a 2D array `psi` by the corresponding value in `shifts_1D`,
        then computes the mean across all shifted rows.

        Parameters
        ----------
        psi : np.ndarray
            2D array of shape (Nθ, Ny)
        shifts_1D : np.ndarray or list 
            1D array of length Nθ, specifying how many pixels to shift each row of `psi`.
            Positive values shift right; negative values shift left.

        Returns
        -------
        shifted_psi : np.ndarray
            2D array of the same shape as `psi`, with each row shifted by the corresponding amount.
        mean_psi : np.ndarray
            1D array of shape (Ny,), representing the mean of the shifted rows over the angles.
        """
        Nth, Ny = psi.shape  
        shifted_psi = np.array([np.roll(psi[theta, :], -shifts_1D[theta]) for theta in range(Nth)])
        mean_psi = np.mean(shifted_psi, axis=0)  # Compute mean along θ
        return shifted_psi, mean_psi

    def calculate_shift_array(self, psi, max_shift, iterations):
        """
        Calculates the shift array iteratively by minimising the error
        for each angle theta.

        Initialize shifts to zero. Processes each angle separately, theta = 0 is the reference.
        Computes the error for each value of the shift range. Finds the value of the
        minimum error. Stores the shift of the minimum error for each theta. The result 
        is refined iteratively. Returns the 1D shift array.

        Parameters
        ----------
        psi : 2D NumPy array
            calculated from M by removing the Legendre polynomials
        max_shift : int
            max pixel shift to search in each direction
        iterations : int
            number of iterations

        Returns:
        --------
        delta_y : 1D NumPy array of size Nth
            optimized shifts per projection
        """
        Nth, Ny = psi.shape
        delta_y_1D = np.zeros(Nth, dtype=int)  
        for _ in range(iterations):  
            for theta in range(1, Nth):  
                shift_range = range(-max_shift, max_shift + 1)
                errors_1D = [self.compute_error(psi, delta_y_1D, theta, scalar_shift) for scalar_shift in shift_range]
                best_shift = shift_range[np.argmin(errors_1D)]  
                delta_y_1D[theta] = best_shift 
        return delta_y_1D
                










