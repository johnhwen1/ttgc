import sys
import os
import numpy as np
from matplotlib import pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ttgc import(helpers, initialize, update, landmark)

def run_sim2d(init_state, cell_position_diffs, 
              n_y, n_x, alpha, beta, I, sigma, T, tau, velocity, start_pos,
              l_pos, l_str, n_ln, l_pinning_n, l_use_nearby, l_lookahead,
              alpha_hebb,
              n_warmup_bins,
              B0, use_GPU, beta_noise=0, weight_noise=0, 
              W_input=None, W_l_input=None, seed=1, output_bins=10000):
    """
    Runs the full simulation given velocity inputs over time.
    
    Args:
        init_state (np.ndarray): array of shape (N,) containing initialized network activity of all grid cells.
        cell_position_diffs (np.ndarray) array of shape (2, N, N), where N = n_y * n_x, containing the differences in positions between pairs of cells. 
        n_y (int): number of grid cells along the y-axis.
        n_x (int): number of grid cells along the x_axis.
        alpha (float): gain.
        beta (float): bias.
        I (float): intensity parameter.
        sigma (float): Gaussian size.
        T (float): shift parameter.
        tau (float): stabilization strength.
        velocity (np.ndarray): array of shape (2,) representing the velocity at a timebin.
        start_pos (np.ndarray): array of shape (2,) representing the starting position of the agent.
        l_pos (np.ndarray): array of shape (n_landmarks, 2) representing the locations of landmarks in the environment. Input np.empty(0) if no landmarks.
        l_str (float): strength of landmark pinning.
        n_ln (int): number of landmark cells per landmark.
        l_pinning_n (int): number of grid cells each landmark cell projects to initially.
        l_use_nearby (bool): whether to use nearby grid cells in setting initial pinning phases.
        l_lookahead (float): the lookahead distance at which landmarks start recruiting their landmark cell(s) activity.
        alpha_hebb (float): Hebbian plasticity gain term.
        n_warmup_bins (int): number of timebins to use with 0 velocity to stabilize a bump of activity through attractor dynamics.
        B0 (np.ndarray): the linear transfer function at the 0th timebin. Input np.zeros(N) if no specific array to be used.
        use_GPU (bool): whether to use GPU.
        beta_noise (float, optional): Gaussian sigma for the amount of noise to be added to the rotation matrix per timebin. 
        weight_noise (float, optional): Gaussian sigma for the amount of noise to be added to grid-grid weight matrix per timebin.
        W_input (np.ndarray, optional): array of shape (n_timebins, N, N) representing the grid-grid weight matrix over time to be used. Useful for replicating simulations with equivalent velocity inputs.
        W_l_input (np.ndarray, optional): array of shape (n_timebins, n_landmarks, n_ln, N) representing the landmark cell to grid weight matrix over time to be used.
        seed (int, optional): for replication.
        output_bins (int, optional): number of timebins before next plot of population activity is rendered during simulation. Set to np.inf to suppress all plots.
    Returns:
        A (np.ndarray): array of shape (n_timebins, N) containing the activities of grid cells over time.
        B (np.ndarray): array of shape (n_timebins, N) containing the linear transfer functions of grid cells over time.
        W (np.ndarray): array of shape (n_timebins, N, N) containing the grid-grid weights over time.
        pos (np.ndarray): array of shape (n_timebins, 2) containing the position of the agent over time.
        A_l (np.ndarray): array of shape (n_timebins, n_landmarks, n_ln) containing the activities of the n_landmark cells over time.
        W_l (np.ndarray): array of shape (n_timebins, n_landmarks, n_ln, N) containing the weights between landmark cells and grid cells over time.
        betas_with_noise (np.ndarray): array of shape (n_timebins,) containing noisy beta values over time.
        weight_noise_vals (np.ndarray): array of shape (n_timebins, N, N) containing the grid-grid noise values added to weights over time.
    TODO:
        Incorporate hebbian plasticity mechanism; make sure to include alpha_decay term.
    """
    alpha_hebb = 0
    n_timebins = velocity.shape[0]
    N = init_state.shape[0]
    A, B, mean_vals, W = initialize.preallocate_sim(n_timebins, N)

    if W_input is not None:
        W = np.copy(W_input)
        betas_with_noise = np.zeros((n_timebins,))
        weight_noise_vals = np.zeros((n_timebins, N, N))
    else:
        W = np.zeros((n_timebins, N, N))
        rng = np.random.default_rng(seed=seed)
        betas_with_noise = beta+np.cumsum(beta_noise*rng.standard_normal(size=((n_timebins,))))
        weight_noise_vals = weight_noise*rng.standard_normal(size=(n_timebins, N, N))

    B[0,:] = B0
    mean_vals[0] = np.mean(B0)

    n_landmarks, l_pos = landmark.get_n_landmarks(l_pos)
    pos = np.cumsum(velocity, axis=0) + start_pos
    A_l = landmark.calc_A_l(pos, l_pos, n_ln, l_lookahead, l_str)

    if W_l_input is not None:
        W_l = W_l_input
    else:
        W_l = np.zeros((n_timebins, n_landmarks, n_ln, N))
        W_l[0] = initialize.initialize_W_l(n_landmarks, n_ln, l_pinning_n, N, l_use_nearby, cell_position_diffs, use_GPU, seed=seed)
        
        if alpha_hebb == 0:
            W_l[1:] = W_l[0]

    if n_warmup_bins > 0:
        A_warmup, B_warmup, mean_vals_warmup, W_warmup = initialize.preallocate_sim(n_warmup_bins, N)
        A_l_warmup = np.zeros((n_warmup_bins, n_landmarks, n_ln))
        W_l_warmup = np.zeros((n_warmup_bins, n_landmarks, n_ln, N))
        
        warmup_velocity = np.tile(np.zeros((1, 2)), (n_warmup_bins, 1))
        warmup_beta = beta * np.ones(n_warmup_bins)
        A_warmup[0, :] = init_state

        for t in np.arange(0, n_warmup_bins-1):
            if t == 0:
                mean_vals_warmup[t] = np.mean(A_warmup[t, :])
            else:
                mean_vals_warmup[t] = np.mean(B_warmup[t, :])
                            
            W_warmup[t, :, :], _, = update.calc_weight_mat(cell_position_diffs, n_y, n_x, alpha, warmup_beta[t], warmup_velocity[t], I, sigma, T, use_GPU)
            A_warmup[t+1, :], B_warmup[t+1, :] = update.update_function(A_warmup[t, :], W_warmup[t, :, :], A_l_warmup[t, :, :].reshape(-1), W_l_warmup[t, :, :, :].reshape(-1, N), 
                                                                        tau, mean_vals_warmup[t])

        init_state = A_warmup[-1, :]
        if not output_bins == np.inf:
            print('Warmup Complete')

    A[0, :] = init_state
    for t in np.arange(0, n_timebins-1):
        if mean_vals[0] == 0:
            mean_vals[t] = np.mean(A[t, :])
        else:
            mean_vals[t] = np.mean(B[t, :])

        # calculate weight matrix and add noise if not supplied
        if W_input is None:  
            if t==0:
                beta_t = beta
            else:
                beta_t = betas_with_noise[t-1]
            W[t, :, :], _ = update.calc_weight_mat(cell_position_diffs, n_y, n_x, alpha, beta_t, velocity[t], I, sigma, T, use_GPU)
            W[t, :, :] += weight_noise_vals[t, :, :] # add noise

        # update next timestep
        A[t+1, :], B[t+1, :] = update.update_function(A[t, :], W[t, :, :], A_l[t, :, :].reshape(-1), W_l[t, :, :, :].reshape(-1, N), tau, mean_vals[t])
        
        if (np.mod(t, output_bins)==0 and not output_bins==np.inf):
            if output_bins > 0:
                print(t)
                fig = plt.figure(figsize=(3,3))
                plt.imshow(A[t, :].reshape(n_x,n_y))
                plt.show()

    return A, B, W, pos, A_l, W_l, betas_with_noise, weight_noise_vals
  
def run_sim1d():
    A = 1
    return A

