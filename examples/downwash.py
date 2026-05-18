"""Model downwash as an external disturbance added to the step pipeline.

Eeach drone creates a cone below itself and applies a downward
force to other drones inside that cone.
"""

import jax.debug as jdbg
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as R
from numpy.typing import NDArray

from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.downwash import (
    hover_induced_velocity,
    jet_centerline_velocity,
    jet_half_width,
    jet_radial_profile,
)
from crazyflow.sim.visualize import draw_line

SIM_DURATION = 4.0
FORCE_ARROW_SCALE = 25.0
FORCE_ARROW_HEAD_LENGTH = 0.08
FORCE_ARROW_HEAD_ANGLE = np.deg2rad(25.0)
SOURCE_POS = np.array([-0.5, -0.5, 1.0])
TARGET_START = np.array([0.5, -0.5, 0.85])
TARGET_END = np.array([-1.5, -0.5, 0.85])


def downwash_fn(data: SimData) -> SimData:
    states = data.states
    pos = states.pos
    quat = states.quat
    mass = data.params.mass
    arm_length = data.params.L
    prop_radius = data.params.prop_radius
    gravity_vec = data.params.gravity_vec

    # Compute the positional differences as a matrix
    pos_differences = pos[:, :, None, :] - pos[:, None, :, :]  # (n_worlds, n_drones, n_drones, 3)

    # Check if the drone is in any of the other drones downwash cone
    # Hortizontal distance from jet centerline
    r = jnp.linalg.norm(pos_differences[..., :2], axis=-1)

    # Positive distance below drones
    s = -pos_differences[..., 2]

    arm_length_source = arm_length[:, None, :, 0] if arm_length.ndim == 3 else arm_length

    # Compute cone border
    cone_border = jet_half_width(motor_distance=arm_length_source, s=s)

    n_drones = pos.shape[1]
    not_self = ~jnp.eye(n_drones, dtype=bool)[None, :, :]  # (n_worlds, n_drones, n_drones)

    in_far_field = (s > 0.0) & (r < cone_border) & not_self

    v_hover = hover_induced_velocity(
        mass=mass,
        gravitational_acceleration=jnp.linalg.norm(gravity_vec),
        air_density=1.225,
        propeller_radius=prop_radius,
        number_propellers=4,
    )
    v_hover_source = v_hover[:, None, :, 0]

    v_center = jet_centerline_velocity(
        hover_induced_velocity=v_hover_source, s=s, motor_distance=arm_length_source
    )

    v_down = jet_radial_profile(
        jet_centerline_velocity=v_center, r=r, jet_half_width=cone_border
    )

    # Mask invalid pairs
    v_down = jnp.where(in_far_field, v_down, 0.0)


    # Add up contributions from all drones
    total_v_down = jnp.sum(v_down, axis=2)  # (n_worlds, n_drones)

    # Create wind vector
    wind_world = jnp.zeros_like(pos)
    wind_world = wind_world.at[..., 2].set(-total_v_down)

    rel_air_world = states.vel - wind_world
    rot = R.from_quat(quat).as_matrix()

    v_a_body = (rot.mT @ rel_air_world[..., None])[..., 0]

    speed = jnp.linalg.norm(v_a_body, axis=-1, keepdims=True)

    # Parasitic drag term
    C_diag = jnp.array([-2.329916287671239e-05, -2.329916287671239e-05, -3.078507303977562e-05])
    parasitic_drag = speed * C_diag * v_a_body

    # Rotor drag term
    K_diag = jnp.array([-2.1991768793537817e-07, -2.1991768793537817e-07, -1.7024656365051572e-07])
    sum_eta = jnp.sum(states.rotor_vel, axis=-1, keepdims=True)
    rotor_drag = sum_eta * K_diag * v_a_body

    force_body = parasitic_drag + rotor_drag
    force_world = (rot @ force_body[..., None])[..., 0]

    jdbg.print("wind_world {x}", x=wind_world)
    jdbg.print("rel_air_world {x}", x=rel_air_world)
    jdbg.print("v_a_body {x}", x=v_a_body)
    jdbg.print("force_body {x}", x=force_body)
    jdbg.print("force_world {x}", x=force_world)

    return data.replace(states=states.replace(force=force_world))


def straight_line_control(sim: Sim, t: float) -> NDArray:
    """Keep drone 0 fixed while drone 1 flies straight underneath it."""
    phase = np.clip(t / SIM_DURATION, 0.0, 1.0)
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    cmd[:, 0, :3] = SOURCE_POS
    cmd[:, 1, :3] = TARGET_START + phase * (TARGET_END - TARGET_START)
    return cmd


def draw_force_arrows(sim: Sim, world: int = 0):
    """Draw the disturbance force at each drone COM."""
    pos = np.array(sim.data.states.pos[world])
    force = np.array(sim.data.states.force[world])
    rgba = np.array([1.0, 0.1, 0.0, 1.0])

    for start, force_world in zip(pos, force):
        arrow = FORCE_ARROW_SCALE * force_world
        arrow_length = np.linalg.norm(arrow)
        if arrow_length < 1e-6:
            continue

        direction = arrow / arrow_length
        end = start + arrow
        draw_line(sim, np.array([start, end]), rgba=rgba, start_size=4.0, end_size=4.0)

        head_length = min(FORCE_ARROW_HEAD_LENGTH, 0.4 * arrow_length)
        side = np.cross(direction, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(side) < 1e-6:
            side = np.array([1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        back = -np.cos(FORCE_ARROW_HEAD_ANGLE) * direction
        side = np.sin(FORCE_ARROW_HEAD_ANGLE) * side

        draw_line(
            sim,
            np.array([end, end + head_length * (back + side)]),
            rgba=rgba,
            start_size=4.0,
            end_size=2.0,
        )
        draw_line(
            sim,
            np.array([end, end + head_length * (back - side)]),
            rgba=rgba,
            start_size=4.0,
            end_size=2.0,
        )


def main(plot: bool = False):
    sim = Sim(n_drones=2, control="state")

    sim.step_pipeline = sim.step_pipeline[:3] + (downwash_fn,) + sim.step_pipeline[3:]
    sim.build_step_fn()

    pos = []
    force = []
    sim.reset()
    sim.render()
    for i in range(int(SIM_DURATION * sim.control_freq)):
        sim.state_control(straight_line_control(sim, i / sim.control_freq))
        sim.step(sim.freq // sim.control_freq)
        current_pos = np.array(sim.data.states.pos[0])
        current_force = np.array(sim.data.states.force[0])
        pos.append(current_pos)
        force.append(current_force)
        draw_force_arrows(sim)
        sim.render()

    sim.close()
    if plot:
        plot_results(pos, force)


def plot_results(pos: list[NDArray], force: list[NDArray]):
    import matplotlib.pyplot as plt  # noqa: F401

    pos = np.array(pos)
    force = np.array(force)
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

    fig_force, ax_force = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    for i, label in enumerate(labels):
        ax_force[i].plot(t, force[:, 0, i], label=f"drone 0 force {label}")
        ax_force[i].plot(t, force[:, 1, i], label=f"drone 1 force {label}", linestyle="--")
        ax_force[i].set_ylabel(f"F{label} [N]")
        ax_force[i].legend()

    fig_force.suptitle("Downwash disturbance force")
    ax_force[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(plot=True)
