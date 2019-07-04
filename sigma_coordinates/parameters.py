"""Control parameters."""

import numpy as np


# Geometry
s_inner = 2.5  # Inner square half-length
s_width = 1  # Channel width
s_n = 64  # Superellipse index

# Discretization
N = 64  # Nominal points-per-unit-length

# Physical
Re = N  # Reynolds number
Pr = 4  # Prandtl number
U = 1  # Imposed velocity
noise_amp = 1e-3  # Initial condition noise amplitude
b_width = 0.04  # Tracer injection width

# Sponge layers
sponge_width = 0.2  # Mask width
sponge_damping = 20  # Damping times across sponge layer

# Timestepping
dt_danger = 0.5  # CFL danger factor
snapshots_dt = 0.1  # Snapshots time cadence
stop_sim_time = 10  # Simulation stop time