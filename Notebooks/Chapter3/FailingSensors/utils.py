from matplotlib import patches
from matplotlib import cm
import numpy as np
from scipy.interpolate import griddata

from mpi4py import MPI
import gmsh
from dolfinx.io import gmshio

def get_msfr_geometry(ax, show_ticks = False, full_core = True):
    # Defining the rectangles for the blue zones
    rect_blanket = patches.Rectangle((1.13, 0.186-2.26 / 2), 0.7, 1.884, linewidth=0.5, facecolor='white', edgecolor='black')

    ax.add_patch(rect_blanket)

    # Setting the limits and aspect
    if full_core:
        ax.set_xlim(0.01, 2.24785852)
        ax.set_ylim(-2.66 / 2, 2.66 / 2)
    else:
        ax.set_xlim(0.01, 2.06450009)
        ax.set_ylim(-2.25500011/2, 2.25500011/2)
    ax.set_aspect('equal')

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

def create_streamlines(domain, velocity):

    x_grid = np.linspace(0, 2.05, 50)
    y_grid = np.linspace(-2.26 / 2, 2.26 / 2, 50)
    X, Y = np.meshgrid(x_grid, y_grid)

    vel_2D = velocity.reshape(-1, 3)

    U_interp = griddata(domain, vel_2D[:,0], (X, Y), method='linear')
    V_interp = griddata(domain, vel_2D[:,2], (X, Y), method='linear')

    return X, Y, U_interp, V_interp


def plot_contour(ax, domain, _snap, 
                 vec_mode_to_plot = None, 
                 cmap = cm.jet, levels = 20,
                 streamline_plot = False, density = 2, linewidth=0.75, 
                 show_ticks = False, full_core = True):
    
    if vec_mode_to_plot is not None:
        if vec_mode_to_plot == 'x':
            snap = _snap[:,::3]
        elif vec_mode_to_plot == 'y':
            snap = _snap[:,1::3]
        elif vec_mode_to_plot == 'z':
            snap = _snap[:,2::3]
        else:
            snap = np.linalg.norm(_snap.reshape(-1, domain.shape[1]), axis=1)
    else:
        snap = _snap

    cont = ax.tricontourf(*domain.T, snap, cmap = cmap, levels = levels)

    # Streamlines
    if streamline_plot:
        
        strm = ax.streamplot(*create_streamlines(domain, _snap), 
                            color='k', density=density, linewidth=linewidth, arrowstyle='->')

    # Add Blanket
    get_msfr_geometry(ax, show_ticks = show_ticks, full_core=full_core)

    return cont