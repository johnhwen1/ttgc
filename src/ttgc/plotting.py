from matplotlib import pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ttgc import(helpers, initialize)

def plot_weight_mat(W, cell_idx, n_y, n_x, three_dim, cmap='magma'):
    '''
    W (np.ndarray): array of shape (N, N) containing the grid-grid weights at a particular time.
    cell_idx (int): reference cell to observe.
    n_y (int): number of grid cells along the y-axis.
    n_x (int): number of grid cells along the x_axis.
    three_dim (bool): whether to plot in 3d.
    cmap (str, optional): colormap.

    '''
    fig = plt.figure(figsize=(n_x/5, n_y/5))

    cell_positions = initialize.calc_cell_positions(n_y, n_x)
    cell_position_diffs = initialize.calc_cell_position_diffs(cell_positions)
    
    xs = cell_position_diffs[0, :, cell_idx]
    ys = cell_position_diffs[1, :, cell_idx]
    zs = W[:,cell_idx]

    if three_dim:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs, ys, zs, c=zs, cmap=cmap)
    else:
        plt.scatter(xs, ys, c=zs, cmap=cmap)
        plt.xticks([])
        plt.yticks([])

    plt.scatter(0, 0, c='r',s=n_x*5,marker='*')
