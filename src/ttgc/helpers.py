import numpy as np
from scipy.signal import correlate2d
import numpy.ma as ma
from scipy.ndimage import rotate

def one_to_two_index(idx, n_y):
    """
    Converts a single index to a two-element list representing the same index.
    
    Args:
        idx (int): single index.
        n_y (int): number of grid cells along the y-axis.
    Returns:
        two_index (list): row#, col#.
    """
    # assuming n_y is dimension 0 (row #), this will gives you a two_index of [row #, col #]
    two_index = []
    two_index.append(int(np.mod(idx, n_y)))
    two_index.append(int(idx - np.mod(idx, n_y))//n_y)
 
    return two_index

def relu(x, threshold=0):
    """
    Rectified linear unit.

    Args:
        x (float): input value.
        threshold (float, optional): rectified value. Defaults to 0.0.       
    Returns:
        float: the greater of x and threshold.
    """
    return np.maximum(x, threshold)
    
def calc_rot_mat(beta):
    """
    Calculates the 2x2 rotation matrix of angle beta (in radians).
    
    Args:
        beta (float): angle.
    Returns:
        R_beta (np.ndarray): array of shape (2, 2) representing the rotation matrix of angle beta.
    """
    R_beta = np.stack([[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]])
                      
    return R_beta

def nonnegative(x):
    """
    Nonnegativity function 
    
    Args:
        x (np.ndarray or float): input.
    Returns:
        nonneg_x (np.ndarray or float): nonnegative version of x.
    """
    nonneg_x = x.clip(min=0)
    return nonneg_x

def str_to_float(arr):
    """
    Convert an array of string values to float values. np.nan values filled where errors occur.
    
    Args:
        arr (np.ndarray): input array with string values. 
    Returns:
        np.ndarray: array with float values.
    """
    convert_arr = []
    
    for s in arr.ravel():    
        try:
            value = float(s)
        except ValueError:
            value = np.nan

        convert_arr.append(value)

    return np.array(convert_arr, dtype=object).reshape(arr.shape)

def calc_2d_maps(A, pos, n_bins):
    """
    Calculates an estimate of 2D rate maps
    
    Args:
        A (np.ndarray): array of shape (n_timebins, N) containing the activities of N cells over time.
        pos (np.ndarray): array of shape (n_timebins, 2) containing the position of the agent over time.
        n_bins (int): number of spatial bins to discretize the environment.
    Returns:
        mean_maps (np.ndarray): array of shape (N, n_bins, n_bins) containing the mean firing activity of each cell.
    """
    n_timebins, N = np.shape(A)
    H = np.digitize(pos, np.linspace(0, np.max(pos), n_bins+1), right=True)
    
    wherebins = []
    for xi in range(n_bins):
        tempx = np.where(H[:, 0] == xi+1)[0]
        
        for yi in range(n_bins):
            tempy = np.where(H[:,1]==yi+1)[0]
    
            wherebins.append(np.intersect1d(tempx,tempy))
    
    mean_maps = np.zeros((N, n_bins, n_bins))
    
    for xi in range(n_bins):
        
        for yi in range(n_bins):
            these_bins = wherebins[xi+yi*n_bins]
            
            if these_bins.size>0:
                mean_maps[:, xi, yi] = np.mean(A[these_bins, :], axis=0)
                
    return mean_maps

def calc_masked_sac(mean_maps, inner_rad, outer_rad):
    """
    Calculates a masked spatial autocorrelogram estimate from 2D rate maps
    
    Args:
        mean_maps (np.ndarray): array of shape (N, n_bins, n_bins) containing the mean firing activity of each cell.
        inner_rad (int): exclusion inner radius.
        outer_rad (int): exclusion outer radius.
    Returns:
        masked_sacs (np.ndarray): array of shape (N, n_spatial_bins*2-1, n_spatial_bins*2-1) containing the masked spatial autocorrelograms of each cell.
        idxs_out_inner0 (np.ndarray): array of variable shape containing indices less than inner radius for axis 0.
        idxs_out_inner1 (np.ndarray): array of variable shape containing indices less than inner radius for axis 1.
        idxs_out_outer0 (np.ndarray): array of variable shape containing indices greater than outer radius for axis 0.
        idxs_out_outer1 (np.ndarray): array of variable shape containing indices greater than outer radius for axis 1.

    """
    N = mean_maps.shape[0]
    n_spatial_bins = mean_maps.shape[-1]

    sacs = np.zeros((N, n_spatial_bins*2-1, n_spatial_bins*2-1))

    for c in range(N):
        sacs[c] = correlate2d(mean_maps[c], mean_maps[c])
    
    mask = np.ones(sacs.shape)

    dist_from_center = np.zeros((2, mask.shape[-1], mask.shape[-1]))
    dist_from_center[0] = np.arange(-mask.shape[-1]//2+1, mask.shape[-1]//2+1)
    dist_from_center[1] = np.arange(-mask.shape[-1]//2+1, mask.shape[-1]//2+1)
    dist_from_center[1] = dist_from_center[1].T
    
    dist_from_center = np.linalg.norm(dist_from_center, axis=0)
    
    idxs_out_inner0 = np.where(dist_from_center<inner_rad)[0]
    idxs_out_inner1 = np.where(dist_from_center<inner_rad)[1]
    idxs_out_outer0 = np.where(dist_from_center>outer_rad)[0]
    idxs_out_outer1 = np.where(dist_from_center>outer_rad)[1]
    
    mask[:, idxs_out_inner0, idxs_out_inner1] = 0
    mask[:, idxs_out_outer0, idxs_out_outer1] = 0
    
    masked_sacs = sacs * mask
    return masked_sacs, idxs_out_inner0, idxs_out_inner1, idxs_out_outer0, idxs_out_outer1

def calc_sxcs(masked_sacs, idxs_out_inner0, idxs_out_inner1, idxs_out_outer0, idxs_out_outer1):
    """
    Calculates a spatial crosscorrelation from the masked autocorrelgrams and rotations of them.
    
    Args:
        masked_sacs (np.ndarray): array of shape (N, n_spatial_bins*2-1, n_spatial_bins*2-1) containing the masked spatial autocorrelograms of each cell.
        idxs_out_inner0 (np.ndarray): array of variable shape containing indices less than inner radius for axis 0.
        idxs_out_inner1 (np.ndarray): array of variable shape containing indices less than inner radius for axis 1.
        idxs_out_outer0 (np.ndarray): array of variable shape containing indices greater than outer radius for axis 0.
        idxs_out_outer1 (np.ndarray): array of variable shape containing indices greater than outer radius for axis 1.
    Returns:
        sxcs (np.ndarray): array of shape (5, N) containing the spatial cross-correlations on rotations of masked sacs and the original one.
    """
    rotations = np.array([60, 120, 30, 90, 150]) # first two are grid angles, the last three are anti-grid
    n_rotations = rotations.size
    N, n_corr_bins, _ = masked_sacs.shape
    
    rotated_masked_sacs = np.zeros((n_rotations, N, n_corr_bins, n_corr_bins))
    sxcs = np.zeros((n_rotations, N))
    
    for r in range(n_rotations):
        this_rotation = rotations[r]

        for c in range(N):
            # first rotate, and then set nans. 
            this_masked_sac = np.copy(masked_sacs[c])
            rotated_masked_sacs[r,c,:,:] = rotate(this_masked_sac, this_rotation, reshape=False)
            this_rotated_masked_sac = np.copy(rotated_masked_sacs[r,c,:,:])

            this_masked_sac[idxs_out_inner0, idxs_out_inner1] = np.nan
            this_masked_sac[idxs_out_outer0, idxs_out_outer1] = np.nan
            this_rotated_masked_sac[idxs_out_inner0, idxs_out_inner1] = np.nan
            this_rotated_masked_sac[idxs_out_outer0, idxs_out_outer1] = np.nan

            sxcs[r,c] = ma.corrcoef(ma.masked_invalid(this_masked_sac.reshape(-1)), 
                                    ma.masked_invalid(this_rotated_masked_sac.reshape(-1)))[0,1]
    return sxcs

def calc_grid_scores(mean_maps, inner_rad, outer_rad):
    """
    Calculates grid scores from mean maps.
    
    Args:
        mean_maps (np.ndarray): array of shape (N, n_bins, n_bins) containing the mean firing activity of each cell.
        inner_rad (int): exclusion inner radius.
        outer_rad (int): exclusion outer radius.
    Returns:
        grid_scores (np.ndarray): array of shape (N,) containing the grid scores of each cell.
    """
    masked_sacs, idxs_out_inner0, idxs_out_inner1, idxs_out_outer0, idxs_out_outer1 = calc_masked_sac(mean_maps, inner_rad, outer_rad)
    sxcs = calc_sxcs(masked_sacs, idxs_out_inner0, idxs_out_inner1, idxs_out_outer0, idxs_out_outer1)
    
    min_vals = np.min(sxcs[0:2], axis=0)
    max_vals = np.max(sxcs[2:], axis=0)
    grid_scores = min_vals - max_vals
    return grid_scores


    