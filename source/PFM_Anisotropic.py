import numpy as np
import ufl
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from mpi4py import MPI
from dolfinx import fem, mesh, default_scalar_type
from petsc4py import PETSc
from dolfinx.io import XDMFFile

#-----------------------------------------------------
# GEOMETRY, MESH SET UP AND FUNCTIONS
#-----------------------------------------------------

# Mesh Generation 
# Mesh size l0 >= 5h

L = 1.0
h = 0.004
num_elements = int(L / h)
domain = mesh.create_unit_square(MPI.COMM_WORLD, num_elements, num_elements)

# Define FunctionSpace one for displacement and other for damage

V_u = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
V_d = fem.functionspace(domain, ("Lagrange", 1))

# Function and Test Functions

u = fem.Function(V_u, name = "Displacement")
v = ufl.TestFunction(V_u)

d = fem.Function(V_d, name = "Damage")
w = ufl.TestFunction(V_d)
d_n = fem.Function(V_d, name = "Damage_initial")

# History field (H) to ensure crack irreversibility

H = fem.Function(V_d, name = "History_Field")

# ---------------------------------------------------
# ADDED PRE-CRACK (Initial Flaw)
# ---------------------------------------------------
# Define a horizontal crack from x=0 to x=0.5 at mid-height (y=0.5)
def initial_crack(x):
    # Returns True for points inside a narrow band representing the crack
    return np.logical_and(x[0] <= L / 2, np.abs(x[1] - L / 2) < 0.004)
    # return np.logical_and(np.abs(x[0] - L / 2) < L/4, np.abs(x[1] - L / 2) < h)

# Locate the nodes (degrees of freedom) in this geometric region
crack_dofs = fem.locate_dofs_geometrical(V_d, initial_crack)

# Set the history field artificially high here so the damage solver instantly forces d=1
H.x.array[crack_dofs] = 1e6

#-----------------------------------------------------
# MATERIAL PARAMETERS
#-----------------------------------------------------

E = fem.Constant(domain, 210000.0)  # Young's Modulus (MPa)
nu = fem.Constant(domain, 0.2)      # Poisson's ratio
Gc = fem.Constant(domain, 2.7)      # Fracture toughness Gc (N/mm)
l0 = fem.Constant(domain, 0.015)     # Lenght Scale Parameter
k_res = fem.Constant(domain, 1e-6)  # Residual Stiffness for stability

# Lame's Parameters

lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))

#-----------------------------------------------------
#   BOUNDARY CONDITIONS
#-----------------------------------------------------

# Fixed bottom (u=0), prescribed displacement at top
def bottom(x):
    return np.isclose(x[1], 0)

def top(x):
    return np.isclose(x[1], L)

def left(x):   
    return np.isclose(x[0], 0)

def right(x):  
    return np.isclose(x[0], 1)

bcs = []

f_dim = domain.topology.dim - 1
bottom_facets = mesh.locate_entities_boundary(domain, f_dim, bottom)
top_facets = mesh.locate_entities_boundary(domain, f_dim, top)
left_facets = mesh.locate_entities_boundary(domain, f_dim, left)
right_facets = mesh.locate_entities_boundary(domain, f_dim, right)

# Fixing the bottom edge

u_zero = np.array([0.0, 0.0], dtype=PETSc.ScalarType)
bc_bottom = fem.dirichletbc(u_zero, fem.locate_dofs_topological(V_u, f_dim, bottom_facets), V_u)

bcs.append(bc_bottom)

# Boundary condition for displacement control at top edge

u_top = fem.Function(V_u)
top_dofs = fem.locate_dofs_topological(V_u, f_dim, top_facets)
bc_top = fem.dirichletbc(u_top, top_dofs)

bcs.append(bc_top)

# Fixing the left, right against vertical displacement

vertical_fixed_facets = np.concatenate([left_facets, right_facets])
#vertical_fixed_facets = right_facets
bc_vertical_fix = fem.dirichletbc(default_scalar_type(0.0), fem.locate_dofs_topological(V_u.sub(1), f_dim, vertical_fixed_facets), V_u.sub(1))

bcs.append(bc_vertical_fix)

# Marking facets at top edge as 1 for Reaction force calculation

marked_values = np.full_like(top_facets, 1, dtype = np.int32)
facet_tags = mesh.meshtags(domain, f_dim, top_facets, marked_values)

metadata = {"quadrature_degree": 4}
ds= ufl.Measure("ds", domain = domain, subdomain_data=facet_tags, metadata = metadata)

#-----------------------------------------------------
#  KINEMATICS AND GOVERNING EQUATIONS
#-----------------------------------------------------

# Kinematics and Isotropic Energy
def epsilon(u):
    return ufl.sym(ufl.grad(u))


eps = epsilon(u)

trace_eps = ufl.tr(eps)

K_o = lmbda + mu 

eps_dev = ufl.dev(eps)

psi_plus = 0.5 * K_o * ufl.max_value(trace_eps, 0)**2 + mu * ufl.inner(eps_dev, eps_dev)

psi_minus = 0.5 * K_o * ufl.min_value(trace_eps, 0)**2
    

I = ufl.Identity(len(u))
# Governing Equations

# --- Displacement Problem (u) ---
# Energetic Degradation Function g(d) = (1-d)^2
g_d = (1.0 - d)**2 + k_res
sigma = g_d * (K_o * ufl.max_value(trace_eps, 0) * I + 2.0 * mu * eps_dev) + K_o * ufl.min_value(trace_eps, 0) * I 

# Virtual Work : Integral (sigma : grad(v))
Pi_u = ufl.inner(sigma, epsilon(v)) * ufl.dx


# --- Damage Problem (d) ---
# Governing Equation : [2 * (1-d)*H / l0] - [Gc/l0 * (1-d)] + [G*l0 * div(grad(d))] = 0

P_d = ( (Gc / l0) * d * w + Gc * l0 * ufl.inner(ufl.grad(d), ufl.grad(w))) * ufl.dx - 2.0 * (1.0 - d) * H * w * ufl.dx

# Create a form for calculation of Reaction force on top

i, j = ufl.indices(2)
n_vec = ufl.FacetNormal(domain)
reaction_form = fem.form(sigma[0,j] * n_vec[j] * ds(1))

#-----------------------------------------------------
# NON-LINEAR SOLVER SETUP
#-----------------------------------------------------

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_monitor": None,
    "snes_atol": 1e-5,
    "snes_rtol": 1e-5,
    "snes_stol": 1e-5,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",    
}

problem_u = NonlinearProblem(
    Pi_u,
    u,
    bcs=bcs,
    petsc_options=petsc_options,
    petsc_options_prefix="Displacement_field",
)

problem_d = NonlinearProblem(
    P_d,
    d,
    bcs= bc,
    petsc_options=petsc_options,
    petsc_options_prefix="Damage_Field",
)

# Staggered Solver and Visulaization

current_psi = fem.Function(V_d)
expr_psi = fem.Expression(psi_plus, V_d.element.interpolation_points)

# Initialize XDMFFile

file_u = XDMFFile(domain.comm, "Displacement_Out_shear_anisotropic.xdmf", "w")
file_d = XDMFFile(domain.comm, "Damage_Out_line_shear_anisotropic.xdmf", "w")

file_u.write_mesh(domain)
file_d.write_mesh(domain)

# Time Stepping Loop (Displacement Control)

t = 0.0
T_final= 125.0

num_steps = 125
delta_t = T_final / num_steps
delta_u = 0.0001

print("Starting Simulation ...")

for step in range(num_steps):
    
    # Update time steps 
    t += delta_t
    
    # Update Boundary Condition
    u_top.interpolate(lambda x: np.stack([np.full(x.shape[1], delta_u * t), np.zeros(x.shape[1])]))
    
    # --- Staggered Solver Iteration ---
                
    # Solve for displacement field
    problem_u.solve()
    
    # Update history filed (H = max(psi_0, H_old))
    current_psi.interpolate(expr_psi)
    H.x.array[:] = np.maximum(H.x.array, current_psi.x.array)
    
    # Solve for damage field
    problem_d.solve()
    
    # Calculate Reaction Force
    # Integrate over the top boundary
    R_x = fem.assemble_scalar(reaction_form)
    
    # For parallel runs, we must sum the values across processors:
    total_R_x = domain.comm.allreduce(R_x, op=MPI.SUM)
    
    # Store values for plotting
    current_disp = delta_u * t
    
    with open("load_displacement_anisotropic.csv", "a") as f:
        if step == 1: f.write("Load,Displacement\n")
        f.write(f"{total_R_x},{current_disp}\n")
           
    # Write results to paraview at each steps
    file_u.write_function(u, t)
    file_d.write_function(d, t)
    

# Save to CSV for plotting later

file_u.close()
file_d.close()

print("Simulation Completed Successfully")
    
    
    
