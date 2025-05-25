import dolfinx
from dolfinx.io import gmshio, XDMFFile
from mpi4py import MPI

import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.integrate import solve_ivp

import ufl
from dolfinx import fem
from dolfinx.fem import (Function, FunctionSpace, Expression, assemble_scalar, form, 
                        locate_dofs_topological, dirichletbc, locate_dofs_geometrical)
from ufl import grad, inner, div, conj, real, imag, Dx, nabla_grad, dot

from pyforce.tools.write_read import StoreFunctionsList as store
from pyforce.tools.backends import norms
from pyforce.tools.functions_list import FunctionsList

from petsc4py import PETSc

from models.N_models import parameterFun as params, steady_neutron_diff_2g, transient_neutron_diff_2g
from models.TH_models import steady_thermal_diffusion, transient_thermal_diffusion

import matplotlib.pyplot as plt

from matplotlib import cm
import pyvista as pv

def plotSnaps(fun, varname, filename, resolution = [2400, 1600]):

    topology, cell_types, geometry = dolfinx.plot.create_vtk_mesh(fun.function_space)

    plotter = pv.Plotter(off_screen=False, border=False, window_size=resolution)
    lab_fontsize = 20
    title_fontsize = 30
    zoom = 1.1

    u_grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    u_grid.point_data[varname] = fun.x.array[:].real
    clim = [min(fun.x.array[:].real), max(fun.x.array[:].real)]
    u_grid.set_active_scalars(varname)
    
    dict_cb = dict(title = varname, width = 0.76,
                    title_font_size=title_fontsize,
                    label_font_size=lab_fontsize,
                    color='k',
                    position_x=0.12, position_y=0.9,
                    shadow=True) 
    
    plotter.add_mesh(u_grid, cmap = cm.jet, clim = clim, show_edges=False, scalar_bar_args=dict_cb)
    plotter.view_xy()
    plotter.camera.zoom(zoom)
    
    plotter.set_background('white', top='white')

    ## Save figure
    plotter.save_graphic(filename+'.pdf')
    # plotter.screenshot(filename+'.png', transparent_background = True,  window_size=resolution)

def extractLine(x, y, domain):

    assert(len(x) == len(y))
    points = np.zeros((3, len(x)))
    points[0, :] = x
    points[1, :] = y
    bb_tree = dolfinx.geometry.BoundingBoxTree(domain, domain.topology.dim)

    cells = []
    points_on_proc = []
    cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, points.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    xPlot = np.array(points_on_proc, dtype=np.float64)

    return xPlot, cells


def transient(t, case = 2):
    if case == 1: # ramp
        return np.piecewise(t, [(t <= 0.2), (t > 0.2)], [lambda t: 1. - 0.11667 * t, lambda t: 0.97666 + 0.0 * t ])
    elif case == 2: # step
        return 0.97666 + 0.0 * t
    elif case == 3: # square
        return np.piecewise(t, [(t <= 0.5), (t > 0.5)], [lambda t: 0.985 + 0.0 * t, lambda t: 1. + 0.0 * t ])
        
def generate_mp_data(path, coupling: dict, D1_reg1: float, finalT: float = 1, dt_save = 0.01, id_case = 1):
    """
    Solving the coupled neutronic (2groups-diffusion)-thermal diffusion using the fully coupled MP model
    

    Parameters
    ----------
    path
        Mother directory in which data as stored (snapshots, figures and .txt data)
    gamma
        Default = 3.034e-3: Feedback coefficients of the absorption XS of group 1
    final_T
        Default = 1.: Final time until the simulation is performed
    dt_save
        Default = 0.01: Saving data every (in time)
    id_case
        Default = 1: Integers defining the kind of transient to solve
                     1: ramp
                     2: step
    """

    ################################ MESH IMPORT ############################################
    gdim = 2

    domain, ct, ft = gmshio.read_from_msh("TWIGL2D.msh", MPI.COMM_WORLD, gdim = gdim)

    domain1_marker = 10
    domain2_marker = 20
    domain3_marker = 30

    ext_marker = 1
    sym_marker = 2

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    ################################ MATERIAL PROPERTIES ############################################
    V_parameters = FunctionSpace(domain, ("DG", 0))
    parameters = params(ct, V_parameters)
    parameters.add(domain1_marker)
    parameters.add(domain2_marker)
    parameters.add(domain3_marker)

    ####### Neutronics parameters #########
    D_1_value = np.array([D1_reg1, 1.4, 1.3])
    D_2_value = np.array([0.4, 0.4, 0.5])

    sigma_a1_value = np.array([0.01, 0.01, 0.008])
    sigma_a2_value = np.array([0.15, 0.15, 0.05])

    nu = 1. 
    nusigma_f1_value  = nu * np.array([0.007, 0.007, 0.003])
    nusigma_f2_value  = nu * np.array([0.2,   0.2,   0.06])

    sigma_s1to2_value = np.array([0.01, 0.01, 0.01]) # thermal to fast

    ####### Thermal parameters #########
    th_cond_value = np.array([8., 1., .5]) # W/m K
    rho_value = np.array([10.45, 10.45, 10.45]) / 1000 # kg / m3
    cp_value  = np.array([235, 235, 235]) / 1000 # kJ/kg K
    rho_cp_value = rho_value * cp_value

    ################################ STEADY STATE SOLUTION (not critical) ############################################

    TD = 600
    Tref = 600
    reactor_power = 1e3

    steady_N  = steady_neutron_diff_2g(domain, ft, 
                                    D_1_value, D_2_value, sigma_a1_value, sigma_a2_value, sigma_s1to2_value, nusigma_f1_value, nusigma_f2_value,
                                    parameters, ext_marker, coupling = coupling)

    steady_TH = steady_thermal_diffusion(domain, ft, 
                                        th_cond_value, rho_cp_value, nusigma_f1_value, nusigma_f2_value,
                                        parameters, ext_marker, TD=TD)
    steady_N.assembleForm()
    steady_TH.assembleForm()

    norm_  = norms(steady_TH.V)

    error = 1.
    ii = 0
    tol = 1e-5
    maxIter = 20

    Tguess = Function(steady_TH.V)
    Tguess.x.set(TD)

    q3_guess = Function(steady_TH.V)

    while error > tol:
        
        # Solve neutron diffusion
        phi_1_ss, phi_2_ss, k_eff = steady_N.solve(Tguess, reactor_power)

        # Solve thermal diffusion
        T_ss = steady_TH.solve(phi_1_ss, phi_2_ss)

        # Compute error
        error_fun_T = Function(steady_TH.V).copy()
        error_fun_T.x.array[:] = T_ss.x.array[:] - Tguess.x.array[:]
        error_T = norm_.L2norm(error_fun_T)

        error_fun_q3 = Function(steady_TH.V).copy()
        error_fun_q3.interpolate(Expression(steady_TH.q3 - q3_guess, steady_TH.V.element.interpolation_points()))
        error_q3 = norm_.L2norm(error_fun_q3)

        error = error_q3 + error_T
        print('Iter #'+str(ii))
        print(f'    Error_T: {error_T :.3e} | error_q3: {error_q3 :.3e} | error: {error :.3e}')

        # Update temperature
        Tguess.x.array[:] = T_ss.x.array[:]
        q3_guess.interpolate(Expression(steady_TH.q3, steady_TH.V.element.interpolation_points()))

        ii += 1

        if error <= tol:
            print('--------')
            print('Converged in '+str(ii)+' iterations, k_eff = {:.6f}'.format(k_eff))
            print('--------')
        if ii > maxIter:
            print('--------')
            print('Warning: maximum iterations limit reached !!!')
            print('--------')
    

    del error, error_fun_q3, error_fun_T, error_q3, error_T, ii, norm_, q3_guess, steady_N, steady_TH, Tguess


    ################################ STEADY STATE SOLUTION (made critical) ############################################
    
    np.savetxt(path+'/k_eff.txt', np.array([k_eff]))

    nu = 1. / k_eff
    nusigma_f1_value  = nu * np.array([0.007, 0.007, 0.003])
    nusigma_f2_value  = nu * np.array([0.2,   0.2,   0.06])

    steady_N  = steady_neutron_diff_2g(domain, ft, 
                                    D_1_value, D_2_value, sigma_a1_value, sigma_a2_value, sigma_s1to2_value, nusigma_f1_value, nusigma_f2_value,
                                    parameters, ext_marker, coupling = coupling)

    steady_TH = steady_thermal_diffusion(domain, ft, 
                                        th_cond_value, rho_cp_value, nusigma_f1_value, nusigma_f2_value,
                                        parameters, ext_marker, TD=TD)
    steady_N.assembleForm()
    steady_TH.assembleForm()

    norm_  = norms(steady_TH.V)

    error = 1.
    ii = 0
    tol = 1e-5
    maxIter = 20

    Tguess = Function(steady_TH.V)
    Tguess.x.set(TD)

    q3_guess = Function(steady_TH.V)

    while error > tol:
        
        # Solve neutron diffusion
        phi_1_ss, phi_2_ss, k_eff = steady_N.solve(Tguess, reactor_power)

        # Solve thermal diffusion
        T_ss = steady_TH.solve(phi_1_ss, phi_2_ss)

        # Compute error
        error_fun_T = Function(steady_TH.V).copy()
        error_fun_T.x.array[:] = T_ss.x.array[:] - Tguess.x.array[:]
        error_T = norm_.L2norm(error_fun_T)

        error_fun_q3 = Function(steady_TH.V).copy()
        error_fun_q3.interpolate(Expression(steady_TH.q3 - q3_guess, steady_TH.V.element.interpolation_points()))
        error_q3 = norm_.L2norm(error_fun_q3)

        error = error_q3 + error_T
        print('Iter #'+str(ii))
        print(f'    Error_T: {error_T :.3e} | error_q3: {error_q3 :.3e} | error: {error :.3e}')

        # Update temperature
        Tguess.x.array[:] = T_ss.x.array[:]
        q3_guess.interpolate(Expression(steady_TH.q3, steady_TH.V.element.interpolation_points()))

        ii += 1

        if error <= tol:
            print('--------')
            print('Converged in '+str(ii)+' iterations, k_eff = {:.6f}'.format(k_eff))
            print('--------')
        if ii > maxIter:
            print('--------')
            print('Warning: maximum iterations limit reached !!!')
            print('--------')
    
    del error, error_fun_q3, error_fun_T, error_q3, error_T, ii, norm_, q3_guess, steady_N, steady_TH, Tguess

    ################################ TRANSIENT  SOLUTION ############################################
    beta_l_value = list()
    beta_l_value.append(np.array([0.0075, 0.0075, 0.0075]))

    beta_tot = beta_l_value[0]

    lambda_p_value = list()
    lambda_p_value.append(np.array([0.08, 0.08, 0.08]))

    velocities = np.array([1.e7, 1.e5])

    trans_N = transient_neutron_diff_2g(domain, ft, 
                                        D_1_value, D_2_value, sigma_a1_value, sigma_a2_value, sigma_s1to2_value, nusigma_f1_value, nusigma_f2_value,
                                        beta_l_value, lambda_p_value, beta_tot, velocities,
                                        parameters, ext_marker, coupling = coupling)
    trans_TH = transient_thermal_diffusion(domain, ft,
                                           th_cond_value, rho_cp_value, nusigma_f1_value, nusigma_f2_value,
                                           parameters, ext_marker, TD=TD)

    sigma_a2_transient_value = lambda tt: np.array([sigma_a2_value[0] * transient(tt, case = id_case), sigma_a2_value[1], sigma_a2_value[2]])

    norm_  = norms(trans_TH.V)

    t = 0.

    dt_value = 1e-4
    trans_N.dt.x.set(dt_value)
    trans_TH.dt.x.set(dt_value)

    # Assembling the variational forms
    trans_N.assembleForm(phi_1_ss, phi_2_ss)
    trans_TH.assembleForm(T_ss)

    t_change = 1e-3
    change = False
    temperature = Function(trans_N.Q)
    temperature.x.array[:] = T_ss.x.array[:]

    power_time = list()
    average_T = list()
    kk = 1
 
    phi_1_t = FunctionsList(trans_N.V.sub(0).collapse()[0])
    phi_2_t = FunctionsList(trans_N.V.sub(1).collapse()[0])
    T_t = FunctionsList(trans_TH.V)

    phi_1_t.append(phi_1_ss)
    phi_2_t.append(phi_2_ss)
    T_t.append(T_ss)
    power_time.append(np.array([t, reactor_power]))
    average_T.append( np.array([t, norm_.average(T_ss)]))

    progress = tqdm(desc="Solving transient coupled neutronics", total=finalT, 
                    bar_format = "{desc}: {percentage:.2f}%|{bar}| {n:.4f}/{total_fmt} [{elapsed}<{remaining}]")

    while t <= finalT:
        t += dt_value

        # Advance in time
        power, phi_1_new, phi_2_new = trans_N.advance(t, sigma_a2_transient_value, temperature)
        T_new, q3 = trans_TH.advance(phi_1_new, phi_2_new)

        # Store temperature and power
        if np.isclose(t - kk * dt_save, 0):
            T_t.append(T_new)
            phi_1_t.append(phi_1_new)
            phi_2_t.append(phi_2_new)
            power_time.append(np.array([t, power]))
            average_T.append(np.array([t, norm_.average(T_new)]))
            kk += 1
            
        # Update Temperature
        temperature.x.array[:] = T_new.x.array[:]

        if ((t >= t_change) & (change == False)):
            dt_value = 1e-3
            trans_N.dt.x.set(dt_value)
            trans_TH.dt.x.set(dt_value)
            trans_TH.A.zeroEntries()
            fem.petsc.assemble_matrix(trans_TH.A, trans_TH.bilinear, trans_TH.bcs)
            trans_TH.A.assemble()  
            change = True
        
        progress.update(dt_value)
        
    num_steps = len(power_time)
    power_list = np.zeros((num_steps, 2))
    average_T_list = np.zeros((num_steps, 2))

    for jj in range(num_steps):
        power_list[jj, 0] = power_time[jj][0]
        power_list[jj, 1] = power_time[jj][1]

        average_T_list[jj, 0] = average_T[jj][0]
        average_T_list[jj, 1] = average_T[jj][1]

    store(domain, phi_1_t, 'phi_1', path+'/phi_1', order = power_list[:, 0])
    store(domain, phi_2_t, 'phi_2', path+'/phi_2', order = power_list[:, 0])
    store(domain, T_t,     'T',     path+'/T',     order = average_T_list[:, 0])

    np.savetxt(path+'/power_time.txt', power_list)
    np.savetxt(path+'/ave_T_time.txt', average_T_list)