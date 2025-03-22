import numpy as np

def calc_cell_positions(n_y, n_x):
    """
    Calculates the positions of each grid cell.
    
    Args:
        n_y (int): number of grid cells along the y-axis.
        n_x (int): number of grid cells along the x_axis.
    Returns:
        cell_positions (np.ndarray): array of shape (2, n_y, n_x) containing the y and x positions of each grid cell.
    """
    cell_positions = np.zeros((2, n_y, n_x))
    cell_positions[0, :, :] = np.tile(np.sqrt(3)/2 * (np.arange(1, n_y+1) - 0.5) / n_y, (n_x, 1)).T
    cell_positions[1, :, :] = np.tile((np.arange(1,n_x+1) - 0.5)/n_x, (n_y, 1))
  
    return cell_positions

def initialize_network(n_y, n_x, seed=0):
    """
    Initializes the network of n_y * n_x grid cells 
    
    Args:
        n_y (int): number of grid cells along the y-axis.
        n_x (int): number of grid cells along the y-axis.
        seed (int, optional): for replication.
    Returns:
        init_state (np.ndarray): array of shape (N,) containing initialized network activity of all grid cells.
    """
    N = n_y * n_x

    rng = np.random.default_rng(seed=seed)
    init_state = rng.uniform(low=0, high=1/np.sqrt(N), size=(N,))
    
    return init_state

def initialize_W_l(n_landmarks, n_ln, l_pinning_n, N, seed=1):
    """
    Generates the landmark cell to grid cell weight matrix. 
    
    Args:
        n_landmarks (int): number of landmarks.
        n_ln (int): number of landmark cells per landmark.
        l_pinning_n (int): number of grid cells each landmark cell projects to initially.
        seed (int, optional): for replication.
    Returns:
        W_l0 (np.ndarray): array of shape (n_landmarks, n_ln, N) containing the initial weight from each landmark cell from each landmark to each grid cell
    """

    rng = np.random.default_rng(seed=seed)
    
    pinning_phases = rng.choice(N, n_landmarks*n_ln*l_pinning_n).reshape(n_landmarks, n_ln, l_pinning_n)
    W_l0 = np.zeros((n_landmarks, n_ln, N))
    # add input to specify landmark pinning phases later 
    # add another conditional here about hebbian plasticity term being 0. otherwise, W_l_input MUST be calculated each timestep
    for l in range(n_landmarks):
        for lc in range(n_ln):
            for li in range(l_pinning_n):
                if li==0:
                    this_cell = pinning_phases[l, lc, li]
                else:
                    if l_use_nearby:
                        cell0 = pinning_phases[l, lc, 0]
                        this_cell_dist_from_all = cell_position_diffs[:, cell0, :]
                        this_cell_tri_dist = calc_tri_dist_squared(this_cell_dist_from_all)
                        closest_cells = np.argsort(this_cell_tri_dist)
                        this_cell = closest_cells[li+1]
                    else:
                        this_cell = pinning_phases[l, lc, li]
    
                W_l0[l, lc, this_cell] = 1/(l_pinning_n*n_ln) # add input to specify landmark pinning weight
    return W_l0

def preallocate_sim(n_timebins, N):
    """
    Preallocate arrays for simulation
    
    Args:
        n_timebins (int): number of timebins.
        N (int): number of grid cells.
    Returns:
        A (np.ndarray): array of shape (n_timebins, N) containing the activities of grid cells over time.
        B (np.ndarray): array of shape (n_timebins, N) containing the linear transfer functions of grid cells over time.
        mean_vals (np.ndarray): array of shape (n_timebins) containing the mean of the linear transfer function over time.
        W (np.ndarray): array of shape (n_timebins, N, N) containing the grid-grid weights over time.
    """
    A = np.zeros((n_timebins, N))
    B = np.zeros((n_timebins, N))
    mean_vals = np.zeros((n_timebins,))
    W = np.zeros((n_timebins, N, N))
                 
    return A, B, mean_vals, W
