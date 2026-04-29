"""Model downwash as an external disturbance added to the step pipeline.

The model is intentionally simple: each drone creates a cone below itself and applies a downward
force to other drones inside that cone.
"""

import jax.numpy as jnp
from numpy.typing import NDArray

from crazyflow.sim import Sim
from crazyflow.sim.data import SimData

DRONE_RADIUS = 0.15
CONE_FACTOR = 0.1
DOWNWASH_HEIGHT = 1.0

def downwash_fn(data: SimData) -> SimData:
    states= data.states
    pos = states.pos
    mass = data.params.mass
    gravity_vec = data.params.gravity_vec

    source = jnp.expand_dims(pos, axis=2)
    target = jnp.expand_dims(pos, axis=1)
    rel = target - source

    z = source[..., 2] - target[..., 2]

    rho = jnp.linalg.norm(rel[..., :2])
    cone_radius = DRONE_RADIUS + CONE_FACTOR * z

    gravity_mag = jnp.linalg.norm(gravity_vec)
    source_weight = jnp.expand_dims(mass, axis=2) * gravity_mag

    cone_radius_sq = jnp.maximum(cone_radius**2, 1e-6)
    radial_profile = (1  - rho**2 / cone_radius_sq)**2
    force_mag = source_weight[..., 0] * jnp.exp(-2.0*z) * radial_profile

    in_cone = (z > 0) & (z < DOWNWASH_HEIGHT) & rho < (cone_radius)
    pair_matrix = jnp.eye(pos.shape[1], dtype=bool)[None, :, :]
    force_mag = jnp.where(in_cone & ~pair_matrix, force_mag, 0.0)

    downwash_z = -jnp.sum(force_mag, axis=1)
    downwash_force = jnp.zeros_like(states.force).at[..., 2].set()
