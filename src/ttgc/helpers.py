import numpy as np

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