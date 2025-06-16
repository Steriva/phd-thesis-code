import numpy as np
from matplotlib import patches, cm

def get_mag_fenicsx(u_vec):
    """
    Function to get the magnitude of a vector field in FEniCSx format (data are assumed 2D).
    """
    u = u_vec[0::2]
    v = u_vec[1::2]
    return np.sqrt(u**2 + v**2)


class PlotFlowCyl():
    
    def __init__(self, domain, centre = (0.2, 0.2), radius = 0.05):
        self.domain = domain

        width = np.max(domain[:,0]) - np.min(domain[:,0])
        height = np.max(domain[:,1]) - np.min(domain[:,1])

        self.aspect = height / width

        self.centre = centre
        self.radius = radius
    
    def create_circle(self, ls=1):
        
        circle = patches.Circle(self.centre, self.radius, edgecolor='black', facecolor='white', linewidth=ls)
        return circle
    
    def plot_contour(self, ax, snap, cmap = cm.RdYlBu_r, levels=40, show_ticks=False):

        if snap.shape[0] == 2*self.domain.shape[0]:
            snap = get_mag_fenicsx(snap)
            
        plot = ax.tricontourf(self.domain[:,0], self.domain[:,1], snap, cmap=cmap, levels=levels)
        ax.add_patch(self.create_circle())
        
        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        return plot
    