import numpy as np
import torch
from helpers import *

if torch.cuda.is_available():
    device = torch.device('cuda')  # Use GPU
    dtype = torch.float32

    # Helper function to convert between numpy arrays and tensors
    to_t = lambda array: torch.tensor(array, device=device, dtype=dtype)
    from_t = lambda tensor: tensor.to("cpu").detach().numpy()

def calc_cell_position_diffs(cell_positions):
    """
    Calculates the position differences between each pair of grid cells.
    
    Args:
        cell_positions (np.ndarray): array of shape (2, n_y, n_x) containing the y and x positions of each grid cell.
    Returns:
        cell_position_diffs (np.ndarray): array of shape (2, N, N), where N = n_y * n_x, containing the differences in positions between pairs of cells. 
                                          Element :,l,m represents the difference between the positions of cells c_l and c_m.
    """
    # if you want the compare cell c_l (at position i1j1) to cell c_m (at position i2j2), you get their 1d indices.
    _, n_y, n_x = cell_positions.shape
    N = n_y * n_x
    cell_position_diffs = np.zeros((2, N, N))

    for i in range(N):
        cell1_idx = one_to_two_index(i, n_y) # sequentially down the rows then left to right across columns
        cell1_pos = cell_positions[:, cell1_idx[0], cell1_idx[1]]
        
        for j in range(N):
            cell2_idx = one_to_two_index(j, n_y)
            cell2_pos = cell_positions[:, cell2_idx[0], cell2_idx[1]]
            cell_position_diffs[:, i, j] = cell1_pos - cell2_pos
            
    return cell_position_diffs

def calc_tri_dist_squared(mat):
    """
    Calculates the square of the tri-norm as defined in Guanella et al. 2007 (see equations 4-13, and the numerator of equation 14).
    
    Args:
        mat (np.ndarray): array of shape (2, N, N) containing the differences in positions between pairs of cells, either with or without velocity modulation.
    Returns:
        tri_dist_squared (np.ndarray): array of shape (2, N, N) containing the tri-norms on the modulated position differences array.
    """
    n_comps = mat.shape[1]
    S = np.array([[0, 0], [-0.5, np.sqrt(3)/2], [-0.5, -np.sqrt(3)/2], 
                  [0.5, np.sqrt(3)/2], [0.5, -np.sqrt(3)/2], [-1, 0], [1, 0]])
    S = np.fliplr(S)
    
    # make copies to do vector subtractions
    S = np.swapaxes(np.expand_dims(S, 1), 2, 0)
    S = np.tile(S, (1, n_comps, 1))
    
    mat = np.expand_dims(mat, 2)
    mat = np.tile(mat, (1, 1, 7))
    sum_mat = S + mat

    if torch.cuda.is_available():
        t_sum_mat = to_t(sum_mat)
        t_tri_dist_squared = torch.min(torch.sum(torch.pow(t_sum_mat, 2), 0), 1)[0]
        tri_dist_squared = from_t(t_tri_dist_squared.cpu())
    else:
        tri_dist_squared = np.min(np.sum(np.power(t_sum_mat, 2), axis=0), axis=1)

    return tri_dist_squared

def calc_weight_mat(cell_position_diffs, n_y, n_x, alpha, beta, velocity, I, sigma, T):
    """
    Calculates the weight matrix used to calculate the contributions of each grid cell to the linear transfer function.
    
    Args:
        cell_position_diffs (np.ndarray): array of shape (2, N, N), where N = n_y * n_x, containing the differences in positions between pairs of cells. 
        n_y (int): number of grid cells along the y-axis.
        n_x (int): number of grid cells along the x_axis.
        alpha (float): gain.
        beta (float): bias.
        velocity (np.ndarray): array of shape (2,) representing the velocity at a timebin.
        I (float): intensity parameter.
        sigma (float): Gaussian size.
        T (float): shift parameter.
    Returns:
        W (np.ndarray): array of shape (N, N) containing grid-cell to grid-cell weights.
        exp_arg (np.ndarray): array of shape (N, N) representing the matrix being exponentiated.  
    """
    N = n_y * n_x
    exp_arg = np.zeros((N, N))

    R_beta = calc_rot_mat(beta)

    modulation_vec = alpha * R_beta @ velocity
    modulation_mat = np.zeros((2, N, N))
    modulation_mat[0,:,:] = modulation_vec[0]
    modulation_mat[1,:,:] = modulation_vec[1]
    
    cell_position_diffs_with_mod = cell_position_diffs + modulation_mat

    modded_mat = cell_position_diffs_with_mod.reshape(2, -1)
    exp_arg = calc_tri_dist_squared(modded_mat).reshape(N, N)
    
    exp_arg = -1*(exp_arg / np.power(sigma, 2))
    
    W = I * np.exp(exp_arg) - T
    return W, exp_arg

def update_function(A_t, W_t, A_l_t, W_l_t, tau, mean_val):
    """
    Updates the grid cell activity vector from one timestep to the next.
    
    Args:
        A_t (np.ndarray): array of shape (N,), where N = n_y * n_x, containing the activity of grid cells at a given timebin.
        W_t (np.ndarray): array of shape (N, N) containing the calculated grid-grid weight matrix at the associated timebin.
        A_l_t (np.ndarray): array of shape (n_landmarks) containing the activity of the landmark cells.
        W_l_t (np.ndarray): array of shape (n_landmarks, N) containing the weights of each landmark to each grid cell.
        tau (float): stabilization strength.
        mean_val (float): the mean of the transfer function in the previous timebin.
    Returns:
        A_t1 (np.ndarray): array of shape (N,), where N = n_y * n_x, containing the activity of grid cells at the next timebin.
        B_t1 (np.ndarray): array of shape (N,), where N = n_y * n_x, containing the linear transfer function at the next timebin.
    TODO:
        expand A_l_t and W_l_t to allow for multiple landmark cells for each landmark.
    """
    B_t1 = A_t + W_t.T @ A_t + W_l_t.T @ A_l_t
    A_t1 = B_t1 + tau * B_t1 * (1 / mean_val - 1)
    A_t1 = nonnegative(A_t1)
    
    return A_t1, B_t1


