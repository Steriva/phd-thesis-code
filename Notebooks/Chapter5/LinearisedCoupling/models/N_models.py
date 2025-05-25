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

class steady_neutron_diff_2g():
    def __init__(self, domain: dolfinx.mesh.Mesh, ft: dolfinx.cpp.mesh.MeshTags_int32, 
                       D_1: np.ndarray, D_2: np.ndarray, sigma_a1: np.ndarray, sigma_a2: np.ndarray, sigma_s1to2: np.ndarray, nusigma_f1: np.ndarray, nusigma_f2: np.ndarray, 
                       param: parameterFun, void_marker: int, coupling: dict):

        self.domain = domain
        self.ft = ft
        self.fdim = domain.geometry.dim - 1
        self.void_marker = void_marker
        self.dx = ufl.Measure("dx", domain=domain)
        self.ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)

        self.P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
        self.V = FunctionSpace(self.domain, ufl.MixedElement([self.P1, self.P1]))
        self.Q = FunctionSpace(self.domain, self.P1)
        self.DG = param.funSpace

        # Parameters at T=T_ref
        self.D1_ref = Function(self.DG)
        self.D2_ref = Function(self.DG)
        self.xs_a1_ref = Function(self.DG)
        self.xs_a2_ref = Function(self.DG)

        param.assign(D_1, self.D1_ref)
        param.assign(D_2, self.D2_ref)
        param.assign(sigma_a1, self.xs_a1_ref)
        param.assign(sigma_a2, self.xs_a2_ref)

        self.xs_f1 = Function(self.DG)
        self.xs_f2 = Function(self.DG)
        self.xs_s1to2 = Function(self.DG)
        param.assign(nusigma_f1, self.xs_f1)
        param.assign(nusigma_f2, self.xs_f2)
        param.assign(sigma_s1to2, self.xs_s1to2)

        # Parameters at T
        self.T = Function(self.Q)
        self.T.x.set(coupling['T_ref'])
        self.T_ref = Function(self.Q)
        self.T_ref.x.set(coupling['T_ref'])
        
        if coupling['mode'] == 'linear':
            # Linear coupling
            intercept = coupling['intercept']
            slope = coupling['slope']
            self.D_1   = self.D1_ref     * (intercept[0] + slope[0] * self.T)
            self.D_2   = self.D2_ref     * (intercept[1] + slope[1] * self.T)
            self.xs_a1 = self.xs_a1_ref  * (intercept[2] + slope[2] * self.T)
            self.xs_a2 = self.xs_a2_ref  * (intercept[3] + slope[3] * self.T)
        else:
            gammas = coupling['gammas']
            # Square Root coupling
            # self.D_1   = self.D1_ref     * (1 + gammas[0] * (ufl.algebra.Power(self.T, 0.5) - ufl.algebra.Power(self.T_ref, 0.5)))
            # self.D_2   = self.D2_ref     * (1 + gammas[1] * (ufl.algebra.Power(self.T, 0.5) - ufl.algebra.Power(self.T_ref, 0.5)))
            # self.xs_a1 = self.xs_a1_ref  * (1 + gammas[2] * (ufl.algebra.Power(self.T, 0.5) - ufl.algebra.Power(self.T_ref, 0.5)))
            # self.xs_a2 = self.xs_a2_ref  * (1 + gammas[3] * (ufl.algebra.Power(self.T, 0.5) - ufl.algebra.Power(self.T_ref, 0.5)))

            # Logarithm coupling
            # self.D_1   = self.D1_ref      * (1 + gammas[0] * ufl.ln(self.T / self.T_ref))
            # self.D_2   = self.D2_ref      * (1 + gammas[1] * ufl.ln(self.T / self.T_ref))
            # self.xs_a1 = self.xs_a1_ref  * (1 + gammas[2] * ufl.ln(self.T / self.T_ref))
            # self.xs_a2 = self.xs_a2_ref  * (1 + gammas[3] * ufl.ln(self.T / self.T_ref))
            
            # Sin coupling
            self.D_1   = self.D1_ref     * (1 + gammas[0] * ufl.sin(2.75 * (self.T - self.T_ref) / self.T_ref))
            self.D_2   = self.D2_ref     * (1 + gammas[1] * ufl.sin(2.75 * (self.T - self.T_ref) / self.T_ref))
            # self.xs_a1 = self.xs_a1_ref  * (1 + gammas[2] * ufl.sin(3 * (self.T - self.T_ref) / self.T_ref))
            # self.xs_a2 = self.xs_a2_ref  * (1 + gammas[3] * ufl.sin(3 * (self.T - self.T_ref) / self.T_ref))

            # Tanh coupling
            # self.D_1   = self.D1_ref     * (1 + gammas[0] * ufl.tanh(2.5 * (self.T - self.T_ref) / self.T_ref))
            # self.D_2   = self.D2_ref     * (1 + gammas[1] * ufl.tanh(2.5 * (self.T - self.T_ref) / self.T_ref))
            self.xs_a1 = self.xs_a1_ref  * (1 + gammas[2] * ufl.tanh(2 * (self.T - self.T_ref) / self.T_ref))
            self.xs_a2 = self.xs_a2_ref  * (1 + gammas[3] * ufl.tanh(2 * (self.T - self.T_ref) / self.T_ref))

        # Trial and test functions
        (self.phi_1,    self.phi_2) = ufl.TrialFunctions(self.V)
        (self.varphi_1, self.varphi_2) = ufl.TestFunctions(self.V)

        # Boundary conditions 
        self.zero = Function(self.V)
        self.zero.x.set(0.)
        self.bc_1 = dirichletbc(self.zero.sub(0), locate_dofs_topological((self.V.sub(0), self.V.sub(0).collapse()[0]), self.fdim, self.ft.find(self.void_marker)), self.V.sub(0))
        self.bc_2 = dirichletbc(self.zero.sub(1), locate_dofs_topological((self.V.sub(1), self.V.sub(1).collapse()[0]), self.fdim, self.ft.find(self.void_marker)), self.V.sub(1))
        
        self.bcs = [self.bc_1, self.bc_2]

        # Old function for the inverse power method
        self.old = Function(self.V)
        self.phi_1_old, self.phi_2_old = self.old.split()
        self.k = Function(self.V.sub(0).collapse()[0])

    def assembleForm(self):

        self.left_side   = ( inner(self.D_1 * grad(self.phi_1), grad(self.varphi_1)) * self.dx + inner( (self.xs_a1 + self.xs_s1to2) * self.phi_1                              , self.varphi_1) * self.dx 
                           + inner(self.D_2 * grad(self.phi_2), grad(self.varphi_2)) * self.dx + inner( (self.xs_a2                ) * self.phi_2  - self.xs_s1to2 * self.phi_1, self.varphi_2) * self.dx )

        self.right_side   = 1. / self.k * inner(self.xs_f1 * self.phi_1_old + self.xs_f2 * self.phi_2_old, self.varphi_1) * self.dx 
        self.right_side2  =               inner(self.xs_f1 * self.phi_1_old + self.xs_f2 * self.phi_2_old, self.varphi_1) * self.dx 

        self.a  = form(self.left_side)
        self.b  = form(self.right_side)
        self.b2 = form(self.right_side2)

        self.A   = fem.petsc.create_matrix(self.a)
        self.rhs = fem.petsc.create_vector(self.b)
        self.B   = fem.petsc.create_vector(self.b2)

        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.A)
        # self.solver.setType(PETSc.KSP.Type.GMRES)
        # self.solver.getPC().setType(PETSc.PC.Type.ILU)   
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)   

    def solve(self, temperature, power = 1, tol = 1e-10, maxIter = 200, LL = 50, verbose = False):

        # Updating temperature
        if len(self.T.x.array[:]) == len(temperature.x.array[:]):
            self.T.x.array[:] = temperature.x.array[:]
        else:
            self.T.interpolate(fem.Expression( temperature, self.Q.element.interpolation_points() ))

        # Assembling LHS matrix
        self.A.zeroEntries()
        fem.petsc.assemble_matrix(self.A, self.a, self.bcs)
        self.A.assemble()  

        # Setting initial guess
        self.phi_1_old.x.set(1.)
        self.phi_2_old.x.set(1.)
        self.k.x.set(1.02)
        k_eff_list = []

        error = 1.
        ii = 0
        new = Function(self.V).copy()
        
        while error > tol: 
            
            # Assembling RHS and applying Dirichlet BC with lifting
            with self.rhs.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.rhs, self.b)
            fem.petsc.apply_lifting(self.rhs, [self.a], [self.bcs])
            self.rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(self.rhs, self.bcs)

            # Solve linear problem
            self.solver.solve(self.rhs, new.vector)
            new.x.scatter_forward()  

            # Computing LHS as vector using new solution
            phi_1_new, phi_2_new = new.split()
            tmpA = fem.petsc.assemble_vector(form(ufl.replace(self.left_side, {self.phi_1: phi_1_new, self.phi_2: phi_2_new})))
            fem.petsc.apply_lifting(tmpA, [self.a], [self.bcs])
            tmpA.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(tmpA, self.bcs)
            Aphi = tmpA[:]

            # Updating old solution
            self.phi_1_old.x.array[:] = phi_1_new.x.array
            self.phi_2_old.x.array[:] = phi_2_new.x.array

            # Computing RHS as vector using new solution
            with self.B.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.B, self.b2)
            fem.petsc.apply_lifting(self.B, [self.a], [self.bcs])
            self.B.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(self.B, self.bcs)
            Bphi = self.B.array[:]
            
            k_eff_list.append( np.dot(Bphi, Bphi) / np.dot(Aphi, Bphi) )
            self.k.x.set(k_eff_list[ii])
            
            if ii > 0:
                error = abs(k_eff_list[ii] - k_eff_list[ii-1]) / k_eff_list[ii]

                if ii % LL == 0 and verbose == True:
                    print(f'    Iter {ii+0:03} | k_eff: {k_eff_list[ii]:.6f} | Rel Error: {error :.3e}')

                if error <= tol and verbose == True:
                    print(f'    Neutronics converged with {ii+0:03} iter | k_eff: {k_eff_list[ii]:.8f} | rho: {(1-1./k_eff_list[ii])*1e5:.2f} pcm | Rel Error: {error :.3e}')
            
            ii += 1

            if ii > maxIter:
                print('Max iteration reached! Exiting loop')
                error = 0

        normalisation = power / assemble_scalar(form( (self.xs_f1 * phi_1_new + self.xs_f2 * phi_2_new) * self.dx) ) 

        phi_1_toAssign = Function(self.V.sub(0).collapse()[0])
        phi_2_toAssign = Function(self.V.sub(1).collapse()[0])
        
        phi_1_toAssign.interpolate(fem.Expression (normalisation * phi_1_new, 
                                                   self.V.sub(0).collapse()[0].element.interpolation_points()) )
        phi_2_toAssign.interpolate(fem.Expression (normalisation * phi_2_new, 
                                                   self.V.sub(1).collapse()[0].element.interpolation_points()) )
                                                   
        return phi_1_toAssign, phi_2_toAssign, k_eff_list[-1]
    

class transient_neutron_diff_2g():
    def __init__(self, domain: dolfinx.mesh.Mesh, ft: dolfinx.cpp.mesh.MeshTags_int32, 
                       D_1: np.ndarray, D_2: np.ndarray, sigma_a1: np.ndarray, sigma_a2: np.ndarray, sigma_s1to2: np.ndarray, nusigma_f1: np.ndarray, nusigma_f2: np.ndarray, 
                       beta_l_value: list, lambda_p_value: list, beta: np.ndarray, veloc: np.ndarray,
                       param: parameterFun, void_marker: int, coupling: dict):

        self.domain = domain
        self.ft = ft
        self.fdim = domain.geometry.dim - 1
        self.void_marker = void_marker
        self.dx = ufl.Measure("dx", domain=domain)
        self.ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)

        assert(len(beta_l_value) == len(lambda_p_value))
        self.prec_groups = len(beta_l_value)

        self.P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
        self.P0 = ufl.FiniteElement("DG", self.domain.ufl_cell(), 0)
        spaces = list()
        spaces.append(self.P1)
        spaces.append(self.P1)
        for ll in range(self.prec_groups):
            spaces.append(self.P0)
        # self.V = FunctionSpace(self.domain, ufl.MixedElement([self.P1, self.P1, self.P0]) )
        self.V = FunctionSpace(self.domain, ufl.MixedElement(spaces) )
        self.Q = FunctionSpace(self.domain, self.P1)
        self.DG = param.funSpace

        # Parameters at T=T_ref
        self.D1_ref = Function(self.DG)
        self.D2_ref = Function(self.DG)
        self.xs_a1_ref = Function(self.DG)
        self.xs_a2_ref = Function(self.DG)

        param.assign(D_1, self.D1_ref)
        param.assign(D_2, self.D2_ref)
        param.assign(sigma_a1, self.xs_a1_ref)
        param.assign(sigma_a2, self.xs_a2_ref)

        self.xs_f1 = Function(self.DG)
        self.xs_f2 = Function(self.DG)
        self.xs_s1to2 = Function(self.DG)
        param.assign(nusigma_f1, self.xs_f1)
        param.assign(nusigma_f2, self.xs_f2)
        param.assign(sigma_s1to2, self.xs_s1to2)

        # Parameters at T
        self.T = Function(self.Q)
        self.T.x.set(coupling['T_ref'])
        self.T_ref = Function(self.Q)
        self.T_ref.x.set(coupling['T_ref'])
        
        if coupling['mode'] == 'linear':
            # Linear coupling
            intercept = coupling['intercept']
            slope = coupling['slope']
            self.D_1   = self.D1_ref     * (intercept[0] + slope[0] * self.T)
            self.D_2   = self.D2_ref     * (intercept[1] + slope[1] * self.T)
            self.xs_a1 = self.xs_a1_ref  * (intercept[2] + slope[2] * self.T)
            self.xs_a2 = self.xs_a2_ref  * (intercept[3] + slope[3] * self.T)
        else:
            gammas = coupling['gammas']
            # Square Root coupling
            # self.D_1   = self.D1_ref     * (1 + gammas[0] * (ufl.algebra.Power(self.T, 0.5) - ufl.algebra.Power(self.T_ref, 0.5)))
            # self.D_2   = self.D2_ref     * (1 + gammas[1] * (ufl.algebra.Power(self.T, 0.5) - ufl.algebra.Power(self.T_ref, 0.5)))
            # self.xs_a1 = self.xs_a1_ref  * (1 + gammas[2] * (ufl.algebra.Power(self.T, 0.5) - ufl.algebra.Power(self.T_ref, 0.5)))
            # self.xs_a2 = self.xs_a2_ref  * (1 + gammas[3] * (ufl.algebra.Power(self.T, 0.5) - ufl.algebra.Power(self.T_ref, 0.5)))

            # Logarithm coupling
            # self.D_1   = self.D1_ref      * (1 + gammas[0] * ufl.ln(self.T / self.T_ref))
            # self.D_2   = self.D2_ref      * (1 + gammas[1] * ufl.ln(self.T / self.T_ref))
            # self.xs_a1 = self.xs_a1_ref  * (1 + gammas[2] * ufl.ln(self.T / self.T_ref))
            # self.xs_a2 = self.xs_a2_ref  * (1 + gammas[3] * ufl.ln(self.T / self.T_ref))
            
            # Sin coupling
            self.D_1   = self.D1_ref     * (1 + gammas[0] * ufl.sin(2.75 * (self.T - self.T_ref) / self.T_ref))
            self.D_2   = self.D2_ref     * (1 + gammas[1] * ufl.sin(2.75 * (self.T - self.T_ref) / self.T_ref))
            # self.xs_a1 = self.xs_a1_ref  * (1 + gammas[2] * ufl.sin(3 * (self.T - self.T_ref) / self.T_ref))
            # self.xs_a2 = self.xs_a2_ref  * (1 + gammas[3] * ufl.sin(3 * (self.T - self.T_ref) / self.T_ref))

            # Tanh coupling
            # self.D_1   = self.D1_ref     * (1 + gammas[0] * ufl.tanh(2.5 * (self.T - self.T_ref) / self.T_ref))
            # self.D_2   = self.D2_ref     * (1 + gammas[1] * ufl.tanh(2.5 * (self.T - self.T_ref) / self.T_ref))
            self.xs_a1 = self.xs_a1_ref  * (1 + gammas[2] * ufl.tanh(2 * (self.T - self.T_ref) / self.T_ref))
            self.xs_a2 = self.xs_a2_ref  * (1 + gammas[3] * ufl.tanh(2 * (self.T - self.T_ref) / self.T_ref))

        # Transient parameters
        self.recip_v1 = fem.Constant(self.domain, PETSc.ScalarType(1./veloc[0]))
        self.recip_v2 = fem.Constant(self.domain, PETSc.ScalarType(1./veloc[1]))

        self.beta_l   = list()
        self.lambda_l = list()

        for ll in range(self.prec_groups):
            tmp = Function(self.DG).copy()
            param.assign(beta_l_value[ll], tmp)
            self.beta_l.append(tmp)

            tmp2 = Function(self.DG).copy()
            param.assign(lambda_p_value[ll], tmp2)
            self.lambda_l.append(tmp2)

        self.beta = Function(self.DG)
        param.assign(beta, self.beta)

        self.param = param

        # Trial and test functions
        self.trial = ufl.TrialFunctions(self.V)
        self.phi_1 = self.trial[0]
        self.phi_2 = self.trial[1]

        self.test = ufl.TestFunctions(self.V)
        self.varphi_1 = self.test[0]
        self.varphi_2 = self.test[1]

        # Boundary conditions 
        self.zero = Function(self.V)
        self.zero.x.set(0.)
        self.bc_1 = dirichletbc(self.zero.sub(0), locate_dofs_topological((self.V.sub(0), self.V.sub(0).collapse()[0]), self.fdim, self.ft.find(self.void_marker)), self.V.sub(0))
        self.bc_2 = dirichletbc(self.zero.sub(1), locate_dofs_topological((self.V.sub(1), self.V.sub(1).collapse()[0]), self.fdim, self.ft.find(self.void_marker)), self.V.sub(1))
        
        self.bcs = [self.bc_1, self.bc_2]

        self.dt = Function(self.DG)
        
        # Old function
        self.old = Function(self.V).copy()
        self.phi_1_old = self.old.sub(0)
        self.phi_2_old = self.old.sub(1)

    def assembleForm(self, phi_1_ss, phi_2_ss):
    
        # Fast Flux
        self.left_side   = dot( (self.recip_v1 / self.dt - (1-self.beta) * self.xs_f1 + self.xs_a1 + self.xs_s1to2) * self.phi_1, self.varphi_1) * self.dx
        self.left_side  += dot( -(1-self.beta) * self.xs_f2 * self.phi_2, self.varphi_1) * self.dx
        self.left_side  += dot( self.D_1 * grad(self.phi_1), grad(self.varphi_1)) * self.dx
        for ll in range(self.prec_groups):
            self.left_side += dot( -self.lambda_l[ll] * self.trial[2+ll], self.varphi_1 ) * self.dx

        self.right_side  = dot(self.recip_v1 / self.dt * self.phi_1_old, self.varphi_1) * self.dx

        # Thermal Flux
        self.left_side  += dot( (self.recip_v2 / self.dt + self.xs_a2) * self.phi_2, self.varphi_2) * self.dx
        self.left_side  += dot( -self.xs_s1to2 * self.phi_1, self.varphi_2) * self.dx
        self.left_side  += dot( self.D_2 * grad(self.phi_2), grad(self.varphi_2)) * self.dx

        self.right_side += dot(self.recip_v2 / self.dt * self.phi_2_old, self.varphi_2) * self.dx

        ## Precursors
        for ll in range(self.prec_groups):
            self.left_side  += dot((1. / self.dt + self.lambda_l[ll]) * self.trial[2+ll] - self.beta_l[ll] * (self.xs_f1 * self.phi_1 + self.xs_f2 * self.phi_2), self.test[2+ll]) * self.dx
            self.right_side += inner(1. / self.dt * self.old.sub(2+ll), self.test[2+ll]) * self.dx
        
        self.bilinear = fem.form(self.left_side)
        self.linear   = fem.form(self.right_side)

        self.A = fem.petsc.create_matrix(self.bilinear)
        self.b = fem.petsc.create_vector(self.linear)

        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.A)
        # self.solver.setType(PETSc.KSP.Type.GMRES)
        # self.solver.getPC().setType(PETSc.PC.Type.ILU)   
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)   

        # Initialise flux and precursors
        self.phi_1_old.interpolate(fem.Expression(phi_1_ss, self.V.sub(0).collapse()[0].element.interpolation_points()))
        self.phi_2_old.interpolate(fem.Expression(phi_2_ss, self.V.sub(1).collapse()[0].element.interpolation_points()))

        for ll in range(self.prec_groups):  
            self.old.sub(2+ll).interpolate(fem.Expression(self.beta_l[ll] / self.lambda_l[ll] * (self.xs_f1 * self.phi_1_old + self.xs_f2 * self.phi_2_old), 
                                                          self.V.sub(2+ll).collapse()[0].element.interpolation_points()))

        # Initialise new solution
        self.new = Function(self.V).copy()
        self.phi_1_new = self.new.sub(0)
        self.phi_2_new = self.new.sub(1)
        self.power_form = form( (self.xs_f1 * self.phi_1_new + self.xs_f2 * self.phi_2_new) * self.dx)

    def advance(self, t: float, sigma_a2_transient_value, temperature: dolfinx.fem.Function):

        # Updating temperature
        if len(self.T.x.array[:]) == len(temperature.x.array[:]):
            self.T.x.array[:] = temperature.x.array[:]
        else:
            self.T.interpolate(fem.Expression( temperature, self.Q.element.interpolation_points() ))

        # assembling LHS matrix
        self.A.zeroEntries()
        self.param.assign(sigma_a2_transient_value(t), self.xs_a2_ref)
        fem.petsc.assemble_matrix(self.A, self.bilinear, self.bcs)
        self.A.assemble()  

        # Update the rhs and apply lifting
        with self.b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b, self.linear)
        fem.petsc.apply_lifting(self.b, [self.bilinear], [self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(self.b, self.bcs)
        
        self.solver.solve(self.b, self.new.vector)
        self.new.x.scatter_forward()

        # Update old
        self.phi_1_old.x.array[:] = self.phi_1_new.x.array[:]
        self.phi_2_old.x.array[:] = self.phi_2_new.x.array[:]

        for ll in range(self.prec_groups):  
            self.old.sub(2+ll).x.array[:] = self.new.sub(2+ll).x.array[:]
        
        # Compute power
        power = assemble_scalar(self.power_form)
        
        return power, self.phi_1_new.collapse(), self.phi_2_new.collapse()

    def solve(self, final_time, change_time, sigma_a2_transient_value, reactor_power = 1., dt_value_eff1 = 1e-6, dt_value_eff2 = 1e-3, LL = 100):

        dt_value_eff = dt_value_eff1
        self.dt.x.set(dt_value_eff)

        # phi_1_t = FunctionsList(self.V.sub(0).collapse()[0])
        # phi_2_t = FunctionsList(self.V.sub(1).collapse()[0])
        # C_1_t   = FunctionsList(self.V.sub(2).collapse()[0])
        # C_2_t   = FunctionsList(self.V.sub(3).collapse()[0])

        t = 0.
        
        ii = 0
        kk = 0
        power_list = []

        change = False

        # progress = tqdm(desc="Solving transient neutronics", total=final_time, 
        #                 bar_format = "{desc}: {percentage:.2f}%|{bar}| {n:.4f}/{total_fmt} [{elapsed}<{remaining}]")

        while t < final_time:
            t += dt_value_eff

            # assembling LHS matrix
            self.A.zeroEntries()
            self.param.assign(sigma_a2_transient_value(t), self.xs_a2_ref)
            fem.petsc.assemble_matrix(self.A, self.bilinear, self.bcs)
            self.A.assemble()  

            # Update the rhs and apply lifting
            with self.b.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.b, self.linear)
            fem.petsc.apply_lifting(self.b, [self.bilinear], [self.bcs])
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(self.b, self.bcs)
            
            # Solve linear problem
            self.solver.solve(self.b, self.new.vector)
            self.new.x.scatter_forward()

            # Update old
            self.phi_1_old.x.array[:] = self.phi_1_new.x.array[:]
            self.phi_2_old.x.array[:] = self.phi_2_new.x.array[:]
            for ll in range(self.prec_groups):  
                self.old.sub(2+ll).x.array[:] = self.new.sub(2+ll).x.array[:]
            
            # Every LL time step
            if ((ii+1) % LL == 0):
                ## Compute power
                tmp_pow = np.array([t, assemble_scalar(self.power_form)])
                power_list.append( tmp_pow )
                print('At t = {:.4e}'.format(t) + ', the power P is {:.6f}'.format(tmp_pow[1] / reactor_power))

                # ## Store fluxes
                # if (t > change_time):
                #     phi_1_t.append(phi_1_new)
                #     phi_2_t.append(phi_2_new)
                # kk += 1

            if ((t > change_time) & (change == False)):

                dt_value_eff = dt_value_eff2
                self.dt.x.set(dt_value_eff)
                self.A.zeroEntries()
                fem.petsc.assemble_matrix(self.A, self.bilinear, self.bcs)
                self.A.assemble()  

                change = True
            
            ii += 1
            # progress.update(dt_value_eff)
        
        ## Compute power for last step
        tmp_pow = np.array([t, assemble_scalar(self.power_form)])
        power_list.append( tmp_pow )
        print('At t = {:.4e}'.format(t) + ', the power P is {:.6f}'.format(tmp_pow[1] / reactor_power))

        # ## Store fluxes
        # phi_1_t.append(phi_1_new)
        # phi_2_t.append(phi_2_new)
        
        num_steps = len(power_list)
        power_time = np.zeros((num_steps, 2))
        
        for jj in range(num_steps):
            power_time[jj, 0] = power_list[jj][0]
            power_time[jj, 1] = power_list[jj][1]

        return power_time