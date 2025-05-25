import dolfinx
import numpy as np
import ufl
from tqdm import tqdm
from dolfinx import fem
from dolfinx.fem import (Function, FunctionSpace, assemble_scalar, form, 
                        locate_dofs_topological, dirichletbc, locate_dofs_geometrical)
from ufl import grad, inner, div, nabla_grad, dot
from petsc4py import PETSc

class parameterFun():
    def __init__(self, cell_tag, funSpace: FunctionSpace):
        self.regionMarker = []
        self.cell_tag = cell_tag
        self.funSpace = funSpace
    
    def add(self, region):
        self.regionMarker.append(region)

    def assign(self, value_vec, fun):
        for idx in range(len(self.regionMarker)):
            regionI = self.regionMarker[idx]
            region_cells = self.cell_tag.find(regionI)
            fun.x.array[region_cells] = np.full_like(region_cells, value_vec[idx], dtype=PETSc.ScalarType)

class steady_thermal_diffusion():
    def __init__(self, domain: dolfinx.mesh.Mesh, ft: dolfinx.cpp.mesh.MeshTags_int32, 
                       th_cond: np.ndarray, rho_cp: np.ndarray, nusigma_f1: np.ndarray, nusigma_f2: np.ndarray,
                       param: parameterFun, void_marker: int, TD: float = 300.):

        self.domain = domain
        self.ft = ft
        self.fdim = domain.geometry.dim - 1
        self.void_marker = void_marker
        self.dx = ufl.Measure("dx", domain=domain)
        self.ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)

        self.finiteElement = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
        self.V = FunctionSpace(self.domain, self.finiteElement)
        self.DG = param.funSpace

        # Parameters
        self.k = Function(self.DG)
        self.rho_cp = Function(self.DG)
        self.xs_f1 = Function(self.DG)
        self.xs_f2 = Function(self.DG)

        param.assign(th_cond, self.k)
        param.assign(rho_cp, self.rho_cp)
        param.assign(nusigma_f1, self.xs_f1)
        param.assign(nusigma_f2, self.xs_f2)

        self.T = ufl.TrialFunction(self.V)
        self.theta = ufl.TestFunction(self.V)
        
        self.TD = Function(self.V)
        self.TD.x.set(TD)
        self.bcs = [dirichletbc(self.TD, locate_dofs_topological(self.V, self.fdim, self.ft.find(self.void_marker)))]
        
    def assembleForm(self):

        # Assembling form to compute the power
        self.phi_1 = Function(self.V)
        self.phi_2 = Function(self.V)
        self.q3 = self.xs_f1 * self.phi_1 + self.xs_f2 * self.phi_2
        
        self.left_side  = dot(self.k / self.rho_cp * grad(self.T), grad(self.theta)) * self.dx
        self.right_side = dot(self.q3 / self.rho_cp, self.theta) * self.dx
        
        self.bilinear  = form(self.left_side)
        self.linear  = form(self.right_side)
        
        self.A = fem.petsc.create_matrix(self.bilinear)
        self.A.zeroEntries()
        fem.petsc.assemble_matrix(self.A, self.bilinear, self.bcs)
        self.A.assemble()  
        self.b = fem.petsc.create_vector(self.linear)

        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.A)
        # self.solver.setType(PETSc.KSP.Type.CG)
        # self.solver.getPC().setType(PETSc.PC.Type.SOR)   
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)   


    def solve(self, phi_1: dolfinx.fem.Function, phi_2: dolfinx.fem.Function):
        
        # Updating fluxes
        if len(self.phi_1.x.array[:]) == len(phi_1.x.array[:]):
            self.phi_1.x.array[:] = phi_1.x.array[:]
        else:
            self.phi_1.interpolate(fem.Expression( phi_1, self.V.element.interpolation_points() ))

        if len(self.phi_2.x.array[:]) == len(phi_2.x.array[:]):
            self.phi_2.x.array[:] = phi_2.x.array[:]
        else:
            self.phi_2.interpolate(fem.Expression( phi_2, self.V.element.interpolation_points() ))

        with self.b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b, self.linear)
        # Apply Dirichlet boundary condition to the vector
        fem.petsc.apply_lifting(self.b, [self.bilinear], [self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(self.b, self.bcs)

        # Solve linear problem
        solution = Function(self.V).copy()
        self.solver.solve(self.b, solution.vector)
        solution.x.scatter_forward()  
                
        return solution
    

class transient_thermal_diffusion():
    def __init__(self, domain: dolfinx.mesh.Mesh, ft: dolfinx.cpp.mesh.MeshTags_int32, 
                       th_cond: np.ndarray, rho_cp: np.ndarray, nusigma_f1: np.ndarray, nusigma_f2: np.ndarray,
                       param: parameterFun, void_marker: int, TD: float = 300.):

        self.domain = domain
        self.ft = ft
        self.fdim = domain.geometry.dim - 1
        self.void_marker = void_marker
        self.dx = ufl.Measure("dx", domain=domain)
        self.ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)

        self.finiteElement = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
        self.V = FunctionSpace(self.domain, self.finiteElement)
        self.DG = param.funSpace

        # Parameters
        self.k = Function(self.DG)
        self.rho_cp = Function(self.DG)
        self.xs_f1 = Function(self.DG)
        self.xs_f2 = Function(self.DG)

        param.assign(th_cond, self.k)
        param.assign(rho_cp, self.rho_cp)
        param.assign(nusigma_f1, self.xs_f1)
        param.assign(nusigma_f2, self.xs_f2)

        self.T = ufl.TrialFunction(self.V)
        self.theta = ufl.TestFunction(self.V)

        self.T_old = Function(self.V)
        self.dt = Function(self.V)
        
        self.TD = Function(self.V)
        self.TD.x.set(TD)
        self.bcs = [dirichletbc(self.TD, locate_dofs_topological(self.V, self.fdim, self.ft.find(self.void_marker)))]
        
    def assembleForm(self, T_ss: dolfinx.fem.Function):

        # Assembling form to compute the power
        # self.phi_1 = Function(self.V)
        # self.phi_2 = Function(self.V)
        self.q3 = Function(self.V) # self.xs_f1 * self.phi_1 + self.xs_f2 * self.phi_2
        
        self.left_side   = dot(1. / self.dt * self.T, self.theta) * self.dx
        self.left_side  += dot(self.k / self.rho_cp * grad(self.T), grad(self.theta)) * self.dx
        
        self.right_side  = dot(1. / self.dt * self.T_old, self.theta) * self.dx
        self.right_side += dot(self.q3 / self.rho_cp, self.theta) * self.dx
        
        self.bilinear  = form(self.left_side)
        self.linear  = form(self.right_side)
        
        self.A = fem.petsc.create_matrix(self.bilinear)
        self.A.zeroEntries()
        fem.petsc.assemble_matrix(self.A, self.bilinear, self.bcs)
        self.A.assemble()  
        self.b = fem.petsc.create_vector(self.linear)

        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.A)
        # self.solver.setType(PETSc.KSP.Type.CG)
        # self.solver.getPC().setType(PETSc.PC.Type.SOR)   
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)   

        # Initialise the old function
        self.T_old.x.array[:] = T_ss.x.array[:]
        self.T_old.x.scatter_forward()

    def advance(self, phi_1: dolfinx.fem.Function, phi_2: dolfinx.fem.Function):

        T_new = Function(self.V).copy()
        
        self.q3.interpolate(fem.Expression(self.xs_f1 * phi_1 + self.xs_f2 * phi_2, self.V.element.interpolation_points()))

        # Updating fluxes
        # if len(self.phi_1.x.array[:]) == len(phi_1.x.array[:]):
        #     self.phi_1.x.array[:] = phi_1.x.array[:]
        # else:
        #     self.phi_1.interpolate(fem.Expression( phi_1, self.V.element.interpolation_points() ))

        # if len(self.phi_2.x.array[:]) == len(phi_2.x.array[:]):
        #     self.phi_2.x.array[:] = phi_2.x.array[:]
        # else:
        #     self.phi_2.interpolate(fem.Expression( phi_2, self.V.element.interpolation_points() ))


        # Assemble RHS with lifting for the Dirichlet BC
        with self.b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b, self.linear)
        fem.petsc.apply_lifting(self.b, [self.bilinear], [self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(self.b, self.bcs)

        # Solve linear problem
        self.solver.solve(self.b, T_new.vector)
        T_new.x.scatter_forward()  
                
        # Update old 
        self.T_old.x.array[:] = T_new.x.array[:]

        return T_new, self.q3