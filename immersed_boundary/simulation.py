

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
b_width = 0.04  # Tracer injection width
# Volume penalization
width_safety = 1  # Mask width safety factor
sponge_damping = 2  # Damping times across sponge layer
# Timestepping
dt_danger = 1/8  # CFL danger factor
snapshots_dt = 0.1  # Snapshots time cadence
stop_sim_time = 40  # Simulation stop time

# Derived parameters
# Geometry
s_outer = s_inner + s_width
r_inner = s_inner
r_outer = np.sqrt(2) * s_outer
# Discretization
N_r = N * 4
N_phi = N * 32 / 4
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
phi_basis = de.Fourier('phi', N_phi, interval=(0, np.pi/2), dealias=3/2)
r_basis = de.Chebyshev('r', N_r, interval=(r_inner, r_outer), dealias=3/2)
domain = de.Domain([phi_basis, r_basis], grid_dtype=np.float64)

# Forcing
phi, r = domain.grids()
x = r * np.cos(phi)
y = r * np.sin(phi)
s = np.maximum(np.abs(x), np.abs(y))
step = lambda x: (1 + special.erf(np.sqrt(np.pi)*x/width)) / 2
# Penalization mask
p_mask = domain.new_field()
p_mask['g'] = step(s-s_outer) + step(s_inner-s)
# Sponge mask
s1 = step(s_width/2 - x) * step(x + s_width / 2) * step(s_outer - y) * step(y - s_inner)
s2 = step(s_width/2 - x) * step(x + s_width / 2) * step(s_outer + y) * step(-y - s_inner)
s3 = step(s_width/2 - y) * step(y + s_width / 2) * step(s_outer - x) * step(x - s_inner)
s4 = step(s_width/2 - y) * step(y + s_width / 2) * step(s_outer + x) * step(-x - s_inner)
s_mask = domain.new_field()
s_mask['g'] = s1 + s2 + s3 + s4
# Reference solution
ref_uphi = domain.new_field()
ref_ur = domain.new_field()
ref_b = domain.new_field()
ref_umag = U * (s_outer - s) * (s - s_inner) / (s_width / 2)**2
ref_umag = np.maximum(ref_umag, 0)
ex_ephi = -np.sin(phi)
ey_ephi = np.cos(phi)
ex_er = np.cos(phi)
ey_er = np.sin(phi)
ref_uphi['g'] = ref_umag * ((s1 - s2) * ex_ephi + (s4 - s3) * ey_ephi)
ref_ur['g'] = ref_umag * ((s1 - s2) * ex_er + (s4 - s3) * ey_er)
s_norm = (s - s_inner) / (s_outer - s_inner)
for si in np.linspace(0,1,7)[1:-1]:
    ref_b['g'] += np.exp(-(s_norm-si)**2/b_width**2)

# Problem
problem = de.IVP(domain, variables=['p','uphi','ur','b','dr_uphi','dr_ur','dr_b'])
problem.parameters['nu'] = nu
problem.parameters['p_tau'] = p_tau
problem.parameters['s_tau'] = s_tau
problem.parameters['p_mask'] = p_mask
problem.parameters['s_mask'] = s_mask
problem.parameters['ref_uphi'] = ref_uphi
problem.parameters['ref_ur'] = ref_ur
problem.parameters['ref_b'] = ref_b
problem.substitutions['Fphi'] = "s_mask/s_tau*(ref_uphi-uphi) - p_mask/p_tau*uphi"
problem.substitutions['Fr'] = "s_mask/s_tau*(ref_ur-ur) - p_mask/p_tau*ur"
problem.substitutions['FB'] = "s_mask/s_tau*(ref_b-b)"
problem.add_equation("ur + r*dr_ur + dphi(uphi) = 0")
problem.add_equation("r**2*dt(uphi) - nu*(r*dr(r*dr_uphi) - uphi + dphi(dphi(uphi)) + 2*dphi(ur)) + r*dphi(p) = -(r**2*ur*dr_uphi + r*uphi*dphi(uphi) + r*ur*uphi) + r**2*Fphi")
problem.add_equation("r**2*dt(ur) - nu*(r*dr(r*dr_ur) - ur + dphi(dphi(ur)) - 2*dphi(uphi)) + r**2*dr(p) = -(r**2*ur*dr_ur + r*uphi*dphi(ur) - r*uphi*uphi) + r**2*Fr")
problem.add_equation("r**2*dt(b) - nu*(r*dr(r*dr(b)) + dphi(dphi(b))) = -(r*uphi*dphi(b) + r**2*ur*dr(b)) + r**2*Fb")
problem.add_equation("dr_ur - dr(ur) = 0")
problem.add_equation("dr_uphi - dr(uphi) = 0")
problem.add_equation("dr_b - dr(b) = 0")
problem.add_bc("left(uphi) = 0")
problem.add_bc("left(ur) = 0")
problem.add_bc("left(dr_b) = 0")
problem.add_bc("right(uphi) = 0")
problem.add_bc("right(ur) = 0", condition="(nphi != 0)")
problem.add_bc("right(dr_b) = 0")
problem.add_bc("right(p) = 0", condition="(nphi == 0)")

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
uphi = solver.state['uphi']
uphi['g'] = noise_amp * noise * (1 - p_mask['g'])

# Integration parameters
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', iter=snapshots_iter, max_writes=10)
snapshots.add_system(solver.state)
snapshots.add_task("(dr(r*uphi) - dphi(ur))/r", name="vorticity")
snapshots.add_task("p_mask")
snapshots.add_task("s_mask")

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
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
