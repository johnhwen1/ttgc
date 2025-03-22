import numpy as np

def get_n_landmarks(l_pos):
    """
    Returns the number of landmarks based on l_pos and creates a pseudo-landmark if l_pos is empty.
    
    Args:
        cell_positions (np.ndarray): array of shape (2, n_y, n_x) containing the y and x positions of each grid cell.
    Returns:
        cell_position_diffs (np.ndarray): array of shape (2, N, N), where N = n_y * n_x, containing the differences in positions between pairs of cells. 
                                          Element :,l,m represents the difference between the positions of cells c_l and c_m.
    """
    if l_pos.size > 0:
        n_landmarks = l_pos.size // 2 # two dimensions in l_pos
    else:
        # create a pseudo-landmark that is never accessible
        l_pos = np.array([-np.inf, -np.inf])
        n_landmarks = 1
    return n_landmarks, l_pos

def calc_A_l(pos, l_pos, n_ln, l_lookahead):
    """
    Calculates the activity of landmark cell(s) based on the agent's position over time and the landmark position(s).
    
    Args:
        pos (np.ndarray): array of shape (n_timebins, 2) representing the agent's 2D position over time.
        l_pos (np.ndarray): array of shape (n_landmarks, 2) representing the n_landmarks 2D position.
        n_ln (int): number of landmark cells per landmark.
        l_lookahead(float): the lookahead distance at which landmarks start recruiting their landmark cell(s) activity.
    Returns:
        A_l (np.ndarray): array of shape (n_timebins, n_landmarks, n_ln) containing the activity of each landmark cell's activity over time.
    """
    n_timebins = pos.shape[0]
    n_landmarks, l_pos = get_n_landmarks(l_pos)
    
    # create pseudo-landmark cell if n_ln set to anything less than 1
    if n_ln < 1:
        A_l = np.zeros((n_timebins, n_landmarks, 1))
    else:    
        pos_all = np.tile(np.expand_dims(pos, (1,2)), (1,n_landmarks,n_ln,1))
        l_pos_all = np.tile(np.expand_dims(l_pos, (0,2)), (n_timebins,1,n_ln,1))
        pos_from_l = pos_all - l_pos_all
        dist_from_l = np.linalg.norm(pos_from_l, axis=-1)
    
        exp_factor = 2 / l_lookahead
        far_idxs = np.abs(dist_from_l) >= l_lookahead # the smaller this becomes, the larger the factor x needs to be below (x * (true_pos_from_l))
        dist_from_l[far_idxs] = -np.inf
        A_l = l_str * np.exp(-np.power(exp_factor*(dist_from_l),2)) 
        
    return A_l
    
