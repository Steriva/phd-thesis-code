from matplotlib import cm

def plot_contour(ax, domain, snap, 
                 cmap = cm.jet, levels = 20,
                 show_ticks = False):
    
    domain = domain[:, :2]
    
    cont = ax.tricontourf(*domain.T, snap, cmap = cmap, levels = levels)

    # Show ticks
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_aspect('equal')
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)

    return cont