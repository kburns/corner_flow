

import numpy as np
from mpi4py import MPI
import time
from scipy import special

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Control parameters
# Geometry
s_inner = 2.5  # Inner square half-length
s_width = 1  # Channel width
# Discretization
N = 64  # Nominal points-per-unit-length
# Physical
Re_danger = 1  # Reynolds number danger factor
U = 1  # Imposed velocity
noise_amp = 1e-3  # Initial condition noise amplitude
# Volume penalization
width_safety = 1  # Mask width safety factor
sponge_damping = 4  # Damping times across sponge layer
# Timestepping
dt_danger = 0.5  # CFL danger factor
snapshots_dt = 0.1  # Snapshots time cadence
stop_sim_time = 10  # Simulation stop time

# Derived parameters
# Geometry
s_outer = s_inner + s_width
# Discretization
N1 = N
N2 = N * 8
# Physical
L = s_width
dx = 1 / N
Re = Re_danger * (L / dx)**(4/3)
nu = L * U / Re
# Volume penalization
width = dx * width_safety
p_tau = width**2 / nu
s_tau = s_width / U / sponge_damping
# Timestepping
dt_cfl = dx / U
dt = dt_danger * min(dt_cfl, p_tau, s_tau)
snapshots_iter = int(np.round(snapshots_dt / dt))

# Bases and domain
x1_basis = de.Fourier('x1', N1, interval=(0, np.pi/2), dealias=3/2)
x2_basis = de.Chebyshev('x2', N2, interval=(0, 1), dealias=3/2)
domain = de.Domain([x1_basis, x2_basis], grid_dtype=np.float64)

# Coordinate mapping
def phi_func(x1, x2):
    return x1
def r_func(x1, x2):
    return 1 + x2
x1, x2 = domain.grids()
phi = domain.new_field()
phi['g'] = phi_func(x1, x2)
d1_phi = phi.differentiate('x1')
d2_phi = phi.differentiate('x2')
r = domain.new_field()
r['g'] = r_func(x1, x2)
d1_r = r.differentiate('x1')
d2_r = r.differentiate('x2')
# X(yi(xi))
#dxi(X) = dyi(X) dxi(yi)

# Forcing
x = r * np.cos(phi)
y = r * np.sin(phi)
s = np.maximum(np.abs(x), np.abs(y))
step = lambda x: (1 + special.erf(np.sqrt(np.pi)*x/width)) / 2
# Sponge mask
s1 = step(s_width/2 - x) * step(x + s_width / 2)
s2 = step(s_width/2 - x) * step(x + s_width / 2)
s3 = step(s_width/2 - y) * step(y + s_width / 2)
s4 = step(s_width/2 - y) * step(y + s_width / 2)
s_mask = domain.new_field()
s_mask['g'] = s1 + s2 + s3 + s4
# Reference solution
ref_uphi = domain.new_field()
ref_ur = domain.new_field()
ref_umag = U * (s_outer - s) * (s - s_inner) / (s_width / 2)**2
ref_umag = np.maximum(ref_umag, 0)
ex_ephi = -np.sin(phi)
ey_ephi = np.cos(phi)
ex_er = np.cos(phi)
ey_er = np.sin(phi)
ref_uphi['g'] = ref_umag * ((s1 - s2) * ex_ephi + (s4 - s3) * ey_ephi)
ref_ur['g'] = ref_umag * ((s1 - s2) * ex_er + (s4 - s3) * ey_er)

# Problem
# Covariant incompressible hydrodynamics
# u1, u2: covariant velocity components
# d1, d2: coordinate derivatives
# D1, D2: covariant derivatives
# gij: contravariant metric components
# g_ij: covariant metric components
# Gi_jk: second-kind Christoffel symbols
problem = de.IVP(domain, variables=['p','u1','u2','d2_u1','d2_u2'])
problem.parameters['nu'] = nu
problem.parameters['s_tau'] = s_tau
problem.parameters['s_mask'] = s_mask
problem.parameters['ref_u1'] = ref_u1
problem.parameters['ref_u2'] = ref_u2
problem.parameters["g11"] = g11
problem.parameters["g12"] = g12
problem.parameters["g21"] = g21
problem.parameters["g22"] = g22
problem.parameters["g_11"] = g_11
problem.parameters["g_12"] = g_12
problem.parameters["g_21"] = g_21
problem.parameters["g_22"] = g_22
problem.substitutions['d1(A)'] = "dx1(A)")
problem.substitutions['d2(A)'] = "dx2(A)")
problem.substitutions['d1_u1'] = "d1(u1)"
problem.substitutions['d1_u2'] = "d1(u2)"
problem.substitutions['f1'] = "s_mask/s_tau*(ref_u1-u1)
problem.substitutions['f2'] = "s_mask/s_tau*(ref_u2-u2)"
for i in ['1','2']:
    for j in ['1','2']:
        for m in ['1','2']:
            problem.substitutions[f'G{m}_{i}{j}'] = "0.5*g1{m}*(d{j}(g_{i}1) + d{i}(g_{j}1) - d1(g_{i}{j})) + 0.5*g2{m}*(d{j}(g_{i}2) + d{i}(g_{j}2) - d2(g_{i}{j}))"
for i in ['1','2']:
    for j in ['1','2']:
        problem.substitutions[f'D{i}_u{j}'] = f"d{i}_u{j} - G1_{i}{j}*u1 - G2_{i}{j}*u2"
for i in ['1','2']:
    for j in ['1','2']:
        for k in ['1','2']:
            problem.substitutions[f'D{i}_D{j}_u{k}'] = f"d{i}(D{j}_u{k}) - G1_{i}{j}*D1_u{k} - G2_{i}{j}*D2_u{k} - G1_{i}{k}*D{j}_u1 - G2_{i}{k}*D{j}_u2"
problem.add_equation("d1_u1 + d2_u2 = g11*D1_u1 + g12*D1_u2 + g21*D2_u1 + g22*D2_u2 - d1(u1) - d2_u2")
problem.add_equation("dt(u1) + d1(p) - nu*(d1(d1_u1) + d2(d2_u1)) = - (g11*u1*D1_u1 + g12*u1*D2_u1 + g21*u2*D1_u1 + g22*u2*D2_u1) + f1 + nu*(g11*D1_D1_u1 + g12*D1_D2_u1 + g21*D2_D1_u1 + g22*D2_D2_u1 - d1(d1_u1) - d2(d2_u1))")
problem.add_equation("dt(u2) + d2(p) - nu*(d1(d1_u2) + d2(d2_u2)) = - (g11*u1*D1_u2 + g12*u1*D2_u2 + g21*u2*D1_u2 + g22*u2*D2_u2) + f2 + nu*(g11*D1_D1_u2 + g12*D1_D2_u2 + g21*D2_D1_u2 + g22*D2_D2_u2 - d1(d1_u2) - d2(d2_u2))")
problem.add_equation("d2_u1 - d2(u1) = 0")
problem.add_equation("d2_u2 - d2(u2) = 0")
problem.add_bc("left(u1) = 0")
problem.add_bc("left(u2) = 0")
problem.add_bc("right(u1) = 0")
problem.add_bc("right(u2) = 0", condition="(nx1 != 0)")
problem.add_bc("right(p) = 0", condition="(nx1 == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
# Velocity noise
noise = rand.standard_normal(gshape)[slices]
u1 = solver.state['u1']
u1['g'] = noise_amp * noise * (1 - p_mask['g'])

# Integration parameters
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', iter=snapshots_iter, max_writes=10)
snapshots.add_system(solver.state)
#snapshots.add_task("(dr(r*uphi) - dphi(ur))/r", name="vorticity")
snapshots.add_task("s_mask")

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
