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
        x (np.ndarray or float): input
    Returns:
        nonneg_x (np.ndarray or float): nonnegative version of x
    """
    nonneg_x = x.clip(min=0)
    return nonneg_x
