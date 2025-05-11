import numpy as np
import matplotlib.pyplot as plt
import math

def pseudo_log(y):
    """Safe log-like transform that handles zero."""
    return np.sign(y) * np.log1p(np.abs(y))

def pseudo_log_inverse(y_log):
    """Inverse of pseudo_log, to convert tick labels back to real units."""
    return np.sign(y_log) * (np.expm1(np.abs(y_log)))

def scatter_each_x_vs_y(x_filen_list, X, Y, cmap='viridis', bins=20):
    if X.shape[1] != len(x_filen_list):
        raise ValueError("Length of x_filen_list must match number of columns in X")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")

    Y_log = pseudo_log(Y)
    N = X.shape[1]
    n_cols = math.ceil(math.sqrt(N))
    n_rows = math.ceil(N / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

    # Sensible ticks and their pseudo-log-transformed positions
    y_real_ticks = [0.3, 1, 3, 10, 30, 100]
    y_log_ticks = pseudo_log(np.array(y_real_ticks))
    for i in range(N):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        hb = ax.hexbin(X[:, i], Y_log, gridsize=bins, cmap=cmap, mincnt=1, linewidths=0.2)

        # In-plot title
        ax.text(0.05, 0.95, x_filen_list[i], ha='left', va='top',
                transform=ax.transAxes, fontsize=10, fontweight='bold')

        # Y-axis formatting
        ax.set_yticks(y_log_ticks)
        if col == 0:
            ax.set_yticklabels([f"{v:.4f}" if v < 0.01 else f"{v:g}" for v in y_real_ticks])
            if row == 2:
                ax.set_ylabel("Burned Area Fraction")
        else:
            ax.set_yticklabels([f"" for v in y_real_ticks])   

        #else:
        #    ax.set_yticks([])

        # Right-hand Y-axis for last column
        if col == n_cols - 1:
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(y_log_ticks)
            ax2.set_yticklabels([f"{v:.4f}" if v < 0.01 else f"{v:g}" for v in y_real_ticks])
            #ax2.set_ylabel("Burned Area Fraction", rotation=270, labelpad=12)
            ax2.yaxis.set_ticks_position('right')
            ax2.yaxis.set_label_position('right')
            
        ax.set_xlabel("X")
        ax.grid(True, linestyle='--', alpha=0.3)

    # Hide unused axes
    for j in range(N, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        fig.delaxes(axes[row][col])

    plt.tight_layout()
    plt.subplots_adjust(right=0.92)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(hb, cax=cbar_ax)
    cbar.set_label("Point Density")
    return fig
