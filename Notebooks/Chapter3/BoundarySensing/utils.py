from matplotlib import patches
from matplotlib import cm
import numpy as np
from scipy.interpolate import griddata

from mpi4py import MPI
import gmsh
from dolfinx.io import gmshio

def get_msfr_geometry(ax, show_ticks = False):
    # Defining the rectangles for the blue zones
    rect_blanket = patches.Rectangle((1.13, 0.186), 0.7, 1.884, linewidth=0.5, facecolor='white', edgecolor='black')

    ax.add_patch(rect_blanket)

    # Setting the limits and aspect
    ax.set_xlim(0.01, 2.06450009)
    ax.set_ylim(0, 2.25500011)
    ax.set_aspect('equal')

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

def create_streamlines(domain, velocity):

    x_grid = np.linspace(0, 2.0645, 50)
    y_grid = np.linspace(0, 2.255, 50)
    X, Y = np.meshgrid(x_grid, y_grid)

    vel_2D = velocity.reshape(-1, 2)

    U_interp = griddata(domain, vel_2D[:,0], (X, Y), method='linear')
    V_interp = griddata(domain, vel_2D[:,1], (X, Y), method='linear')

    # Mask the blanket region
    blanket_x0, blanket_y0 = 1.13, 0.186
    blanket_w, blanket_h = 0.7, 1.884

    mask = (X > blanket_x0) & (X < blanket_x0 + blanket_w) & \
           (Y > blanket_y0) & (Y < blanket_y0 + blanket_h)

    U_interp[mask] = np.nan
    V_interp[mask] = np.nan

    return X, Y, U_interp, V_interp

def plot_contour(ax, domain, _snap, 
                 vec_mode_to_plot = None, 
                 cmap = cm.jet, levels = 20,
                 streamline_plot = False, density = 2, linewidth=0.75, show_ticks = False):
    
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
    get_msfr_geometry(ax, show_ticks = show_ticks)

    return cont

def create_mesh_dolfinx(path):

    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    gdim = 2
     
    # Initialize the gmsh module
    gmsh.initialize()

    # Load the .geo file
    gmsh.merge(path)
    gmsh.model.geo.synchronize()

    # Set algorithm (adaptive = 1, Frontal-Delaunay = 6)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    # Linear Finite Element
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.optimize("Netgen")

    # Import into dolfinx
    domain, ct, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim = gdim )

    # Finalize the gmsh module
    gmsh.finalize()
        
    ft.name = "Facet markers"

    domain.topology.create_connectivity(gdim, gdim)
    domain.topology.create_connectivity(gdim-1, gdim)
    
    return domain, ct, ft, gdim