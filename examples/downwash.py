"""Model downwash as an external disturbance added to the step pipeline.

Eeach drone creates a cone below itself and applies a downward
force to other drones inside that cone.
"""

import jax.debug as jdbg
import jax.numpy as jnp
import numpy as np
import numpy as np
from numpy.typing import NDArray

from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.visualize import draw_line
from crazyflow.sim.visualize import draw_line

DRONE_RADIUS = 0.08
DRONE_RADIUS = 0.08
CONE_FACTOR = 0.1
DOWNWASH_HEIGHT = 1.0

SIM_DURATION = 8.0
SOURCE_POS = np.array([0.0, 0.0, 1.0])
TARGET_START = np.array([-0.4, 0.0, 0.85])
TARGET_END = np.array([0.4, 0.0, 0.85])

CONE_SEGMENTS = 5
CONE_LEVELS = (0.0, 0.5, 1.0)
CONE_COLOR = np.array([0.1, 0.55, 1.0, 0.35])


def downwash_fn(data: SimData) -> SimData:
    states = data.states
    states = data.states
    pos = states.pos
    quat = states.quat
    mass = data.params.mass
    arm_length = data.params.L
    gravity_vec = data.params.gravity_vec

    source = jnp.expand_dims(pos, axis=2)
    target = jnp.expand_dims(pos, axis=1)

    rel = target - source

    z = source[..., 2] - target[..., 2]  # (world, source drone, target drone, rotor)

    rho = jnp.linalg.norm(rel[..., :2], axis=-1)
    cone_radius = DRONE_RADIUS + CONE_FACTOR * z

    gravity_mag = jnp.linalg.norm(gravity_vec)
    source_weight = mass[..., 0][:, :, None, None] * gravity_mag / 10
    cone_radius_sq = jnp.maximum(cone_radius**2, 1e-6)
    radial_profile = (1 - rho**2 / cone_radius_sq) ** 2
    force_mag = source_weight[..., 0] * jnp.exp(-2.0 * z) * radial_profile

    in_cone = (z > 0) & (z < DOWNWASH_HEIGHT) & (rho < cone_radius)
    pair_matrix = jnp.eye(pos.shape[1], dtype=bool)[None, :, :]
    force_mag = jnp.where(in_cone & ~pair_matrix, force_mag, 0.0)

    downwash_z = -jnp.sum(force_mag, axis=1)
    downwash_force = jnp.zeros_like(states.force).at[..., 2].set(downwash_z)
    jdbg.print("Downwash force {x}", x=downwash_force)

    return data.replace(states=states.replace(force=downwash_force))


def straight_line_control(sim: Sim, t: float) -> NDArray:
    """Keep drone 0 fixed while drone 1 flies straight underneath it."""
    phase = np.clip(t / SIM_DURATION, 0.0, 1.0)
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    cmd[:, 0, :3] = SOURCE_POS
    cmd[:, 1, :3] = TARGET_START + phase * (TARGET_END - TARGET_START)
    return cmd


def draw_downwash_cones(sim: Sim, positions: NDArray):
    """Draw sparse wireframe downwash cones below each drone."""
    theta = np.linspace(0.0, 2.0 * np.pi, CONE_SEGMENTS + 1)
    unit_circle = np.column_stack((np.cos(theta), np.sin(theta)))

    for pos in positions:
        rings = []
        for level in CONE_LEVELS:
            depth = DOWNWASH_HEIGHT * level
            radius = DRONE_RADIUS + CONE_FACTOR * depth
            ring = np.empty((CONE_SEGMENTS + 1, 3))
            ring[:, :2] = pos[:2] + radius * unit_circle
            ring[:, 2] = pos[2] - depth
            rings.append(ring)
            draw_line(sim, ring, rgba=CONE_COLOR, start_size=0.006, end_size=0.006)

        top_ring, bottom_ring = rings[0], rings[-1]
        for i in range(CONE_SEGMENTS):
            draw_line(
                sim,
                np.stack((top_ring[i], bottom_ring[i])),
                rgba=CONE_COLOR,
                start_size=0.004,
                end_size=0.004,
            )


def main(plot: bool = False):
    sim = Sim(n_drones=2, control="state")

    sim.step_pipeline = sim.step_pipeline[:3] + (downwash_fn,) + sim.step_pipeline[3:]
    sim.build_step_fn()

    pos = []
    sim.reset()
    initial_pos = sim.data.states.pos.at[0, 0].set(SOURCE_POS)
    initial_pos = initial_pos.at[0, 1].set(TARGET_START)
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=initial_pos))
    sim.render()
    for i in range(int(SIM_DURATION * sim.control_freq)):
        sim.state_control(straight_line_control(sim, i / sim.control_freq))
        sim.step(sim.freq // sim.control_freq)
        current_pos = np.array(sim.data.states.pos[0])
        pos.append(current_pos)
        draw_downwash_cones(sim, current_pos)
        sim.render()

    sim.close()
    if plot:
        plot_results(pos)


def plot_results(pos: list[NDArray]):
    import matplotlib.pyplot as plt  # noqa: F401

    pos = np.array(pos)
    t = np.linspace(0, SIM_DURATION, len(pos))

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    labels = ("x", "y", "z")
    for i, label in enumerate(labels):
        ax[i].plot(t, pos[:, 0, i], label=f"drone 0 {label}")
        ax[i].plot(t, pos[:, 1, i], label=f"drone 1 {label}", linestyle="--")
        ax[i].set_ylabel(label)
        ax[i].legend()

    fig.suptitle("Two-drone downwash example")
    ax[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(plot=True)
