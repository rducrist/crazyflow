"""Model downwash as an external disturbance added to the step pipeline.

Eeach drone creates a cone below itself and applies a downward
force to other drones inside that cone.
"""

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.visualize import draw_line

DRONE_RADIUS = 0.08
CONE_FACTOR = 0.1
DOWNWASH_HEIGHT = 1.0
DOWNWASH_FORCE_SCALE = 1.0
DOWNWASH_TORQUE_SCALE = 1.0

SIM_DURATION = 8.0
SOURCE_POS = np.array([-0.5, -0.5, 1.0])
TARGET_START = np.array([0.5, -0.5, 0.85])
TARGET_END = np.array([-1.5, -0.5, 0.85])

CONE_SEGMENTS = 5
CONE_LEVELS = (0.0, 0.5, 1.0)
CONE_COLOR = np.array([0.1, 0.55, 1.0, 0.35])

ROTOR_LAYOUT = jnp.array(
    [[1.0, -1.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
) / jnp.sqrt(2.0)


def downwash_fn(data: SimData) -> SimData:
    states = data.states
    pos = states.pos
    quat = states.quat
    mass = data.params.mass
    arm_length = data.params.L
    gravity_vec = data.params.gravity_vec

    rotor_offsets_world = rotor_offsets(arm_length, quat)
    rotor_pos = pos[..., None, :] + rotor_offsets_world

    source = pos[:, :, None, None, :]
    target = rotor_pos[:, None, :, :, :]

    rel = target - source

    z = source[..., 2] - target[..., 2]  # (world, source drone, target drone, rotor)

    rho = jnp.linalg.norm(rel[..., :2], axis=-1)
    cone_radius = DRONE_RADIUS + CONE_FACTOR * z

    gravity_mag = jnp.linalg.norm(gravity_vec)
    source_weight = mass[..., 0][:, :, None, None] * gravity_mag / 10
    cone_radius_sq = jnp.maximum(cone_radius**2, 1e-6)
    radial_profile = (1 - rho**2 / cone_radius_sq) ** 2
    force_mag = source_weight * jnp.exp(-2.0 * z) * radial_profile

    in_cone = (z > 0) & (z < DOWNWASH_HEIGHT) & (rho < cone_radius)
    pair_matrix = jnp.eye(pos.shape[1], dtype=bool)[None, :, :, None]
    force_mag = jnp.where(in_cone & ~pair_matrix, force_mag, 0.0)

    rotor_force_z = -jnp.sum(force_mag, axis=1) / ROTOR_LAYOUT.shape[0]
    rotor_forces = jnp.zeros_like(rotor_pos).at[..., 2].set(rotor_force_z)
    downwash_force = DOWNWASH_FORCE_SCALE * jnp.sum(rotor_forces, axis=2)
    downwash_torque = DOWNWASH_TORQUE_SCALE * jnp.sum(
        jnp.cross(rotor_offsets_world, rotor_forces, axis=-1), axis=2
    )

    return data.replace(states=states.replace(force=downwash_force, torque=downwash_torque))


def rotor_offsets(arm_length: jnp.ndarray, quat: jnp.ndarray) -> jnp.ndarray:
    """Return x-configuration rotor offsets from each drone's COM in world frame."""
    if arm_length.ndim == 0:
        offsets_body = arm_length * ROTOR_LAYOUT
        offsets_body = jnp.broadcast_to(offsets_body, (*quat.shape[:-1], *ROTOR_LAYOUT.shape))
    else:
        if arm_length.shape[-1] == 1:
            arm_length = arm_length[..., 0]
        offsets_body = arm_length[..., None, None] * ROTOR_LAYOUT

    return rotate_body_to_world(offsets_body, quat)


def rotate_body_to_world(vectors: jnp.ndarray, quat: jnp.ndarray) -> jnp.ndarray:
    """Rotate vectors by xyzw quaternions."""
    q_vec = quat[..., None, :3]
    q_w = quat[..., None, 3:4]
    return vectors + 2.0 * jnp.cross(
        q_vec, jnp.cross(q_vec, vectors, axis=-1) + q_w * vectors, axis=-1
    )


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
    sim = Sim(n_drones=2, control="state", physics="first_principles")

    sim.step_pipeline = sim.step_pipeline[:3] + (downwash_fn,) + sim.step_pipeline[3:]
    sim.build_step_fn()

    pos = []
    quat = []
    force = []
    torque = []
    sim.reset()
    sim.render()
    for i in range(int(SIM_DURATION * sim.control_freq)):
        sim.state_control(straight_line_control(sim, i / sim.control_freq))
        sim.step(sim.freq // sim.control_freq)
        current_pos = np.array(sim.data.states.pos[0])
        current_quat = np.array(sim.data.states.quat[0])
        current_force = np.array(sim.data.states.force[0])
        current_torque = np.array(sim.data.states.torque[0])
        pos.append(current_pos)
        quat.append(current_quat)
        force.append(current_force)
        torque.append(current_torque)
        draw_downwash_cones(sim, current_pos)
        sim.render()

    sim.close()
    if plot:
        plot_results(pos, quat, force, torque)


def plot_results(
    pos: list[NDArray], quat: list[NDArray], force: list[NDArray], torque: list[NDArray]
):
    import matplotlib.pyplot as plt  # noqa: F401
    from scipy.spatial.transform import Rotation as R

    pos = np.array(pos)
    quat = np.array(quat)
    force = np.array(force)
    torque = np.array(torque)
    rpy_drone_1 = R.from_quat(quat[:, 1]).as_euler("xyz", degrees=True)
    t = np.linspace(0, SIM_DURATION, len(pos))

    fig, ax = plt.subplots(3, 3, sharex=True, figsize=(14, 6))
    pos_labels = ("x", "y", "z")
    rpy_labels = ("roll", "pitch", "yaw")
    for i, label in enumerate(pos_labels):
        ax[i, 0].plot(t, pos[:, 0, i], label=f"drone 0 {label}")
        ax[i, 0].plot(t, pos[:, 1, i], label=f"drone 1 {label}", linestyle="--")
        ax[i, 0].set_ylabel(f"{label} [m]")
        ax[i, 0].legend()

        ax[i, 1].plot(t, rpy_drone_1[:, i], label=f"drone 1 {rpy_labels[i]}")
        ax[i, 1].set_ylabel(f"{rpy_labels[i]} [deg]")
        ax[i, 1].legend()

        ax[i, 2].plot(t, force[:, 1, i], label=f"force {label}")
        ax[i, 2].plot(t, torque[:, 1, i], label=f"torque {label}", linestyle="--")
        ax[i, 2].set_ylabel("N / Nm")
        ax[i, 2].legend()

    fig.suptitle("Two-drone downwash example")
    ax[-1, 0].set_xlabel("Time (s)")
    ax[-1, 1].set_xlabel("Time (s)")
    ax[-1, 2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(plot=True)
