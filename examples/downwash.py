"""Model downwash as an external disturbance added to the step pipeline.

Eeach drone creates a cone below itself and applies a downward
force to other drones inside that cone.
"""

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
DOWNWASH_CONE_DEPTH = 0.8
DOWNWASH_CONE_RINGS = 8
DOWNWASH_CONE_SEGMENTS = 48
DOWNWASH_CONE_RAYS = 8
SOURCE_POS = np.array([-0.5, -0.5, 1.0])
TARGET_START = np.array([0.5, -0.5, 0.65])
TARGET_END = np.array([-1.5, -0.5, 0.65])


def downwash_fn(data: SimData) -> SimData:
    states = data.states
    pos = states.pos
    quat = states.quat
    mass = data.params.mass
    arm_length = data.params.L
    prop_radius = data.params.prop_radius
    gravity_vec = data.params.gravity_vec
    mixing_matrix = data.params.mixing_matrix

    rot = R.from_quat(quat).as_matrix()

    # mixing_matrix = [
    # [-1.0, -1.0,  1.0,  1.0],
    # [-1.0,  1.0,  1.0, -1.0],
    # [-1.0,  1.0, -1.0,  1.0]]

    # We can infer the positions of the rotors from the mixing matrix
    rotor_offsets_body = (arm_length / jnp.sqrt(2)) * jnp.stack(
        (
            -mixing_matrix[..., 1, :],
            mixing_matrix[..., 0, :],
            jnp.zeros_like(mixing_matrix[..., 0, :]),
        ),
        axis=-1,
    )

    rotor_offsets_world = (rot[..., None, :, :] @ rotor_offsets_body[..., :, None])[..., 0]
    rotor_pos_world = pos[..., None, :] + rotor_offsets_world  # (n_worlds, n_drones, n_rotors, 3)

    # Compute the positional differences as a matrix
    pos_differences = rotor_pos_world[:, :, :, None, :] - pos[:, None, None, :, :]
    # (n_worlds, target_drone, target_rotor, source_drone, 3)

    # Check if the drone is in any of the other drones downwash cone
    # Hortizontal distance from jet centerline
    r = jnp.linalg.norm(pos_differences[..., :2], axis=-1)

    # Positive distance below drones
    s = -pos_differences[..., 2]

    motor_distance_source = 2.0 * arm_length

    # Compute cone border
    cone_border = jet_half_width(motor_distance=motor_distance_source, s=s)

    n_drones = pos.shape[1]
    not_self = ~jnp.eye(n_drones, dtype=bool)[
        None, :, None, :
    ]  # (n_worlds, target_drone, target_rotor, source_drone)

    in_far_field = (s > 0.0) & (r < cone_border) & not_self

    # Still stand air velocity for every drone
    v_hover = hover_induced_velocity(
        mass=mass,
        gravitational_acceleration=jnp.linalg.norm(gravity_vec),
        air_density=1.225,
        propeller_radius=prop_radius,
        number_propellers=4,
    )
    v_hover_source = v_hover[:, None, None, :, 0]

    v_center = jet_centerline_velocity(
        hover_induced_velocity=v_hover_source, s=s, motor_distance=motor_distance_source
    )

    v_down = jet_radial_profile(jet_centerline_velocity=v_center, r=r, jet_half_width=cone_border)

    # Mask invalid pairs
    v_down = jnp.where(in_far_field, v_down, 0.0)

    # Add up source drone contributions at each target rotor.
    total_v_down_per_rotor = jnp.sum(v_down, axis=-1)
    total_v_down = jnp.mean(total_v_down_per_rotor, axis=-1, keepdims=True)

    # Adjust for air above propeller moving at wind speed created by downwash
    v_hover_adjusted = total_v_down / 2 + jnp.sqrt((total_v_down / 2) ** 2 + v_hover**2)

    v_hover_source = v_hover_adjusted[:, None, None, :, 0]

    v_center = jet_centerline_velocity(
        hover_induced_velocity=v_hover_source, s=s, motor_distance=motor_distance_source
    )

    v_down = jet_radial_profile(jet_centerline_velocity=v_center, r=r, jet_half_width=cone_border)

    v_down = jnp.where(in_far_field, v_down, 0.0)
    total_v_down_per_rotor = jnp.sum(v_down, axis=-1)

    # Create rotor-local wind vectors for rotor drag and torque.
    wind_world_rotor = jnp.zeros_like(rotor_pos_world)
    wind_world_rotor = wind_world_rotor.at[..., 2].set(-total_v_down_per_rotor)

    target_z_world = rot[..., :, 2]

    rel_air_world_rotor = states.vel[..., None, :] - wind_world_rotor

    v_a_body_rotor = (rot.mT[..., None, :, :] @ rel_air_world_rotor[..., None])[..., 0]
    v_a_body_still = (rot.mT @ states.vel[..., None])[..., 0]

    # Parasitic drag term
    total_v_down_com = -jnp.mean(
        jnp.sum(wind_world_rotor * target_z_world[..., None, :], axis=-1), axis=-1
    )

    wind_world = jnp.zeros_like(pos)
    wind_world = wind_world.at[..., 2].set(-total_v_down_com)
    rel_air_world = states.vel - wind_world
    v_a_body = (rot.mT @ rel_air_world[..., None])[..., 0]
    speed = jnp.linalg.norm(v_a_body, axis=-1, keepdims=True)
    speed_still = jnp.linalg.norm(v_a_body_still, axis=-1, keepdims=True)

    C_diag = jnp.array([-2.329916287671239e-05, -2.329916287671239e-05, -3.078507303977562e-05])
    parasitic_drag = speed * C_diag * v_a_body
    parasitic_drag_still = speed_still * C_diag * v_a_body_still

    # Rotor drag term
    K_diag = jnp.array([-2.1991768793537817e-07, -2.1991768793537817e-07, -1.7024656365051572e-07])
    rotor_vels_body = v_a_body_rotor + jnp.cross(
        states.ang_vel[..., None, :], rotor_offsets_body
    )
    rotor_vels_body_still = v_a_body_still[..., None, :] + jnp.cross(
        states.ang_vel[..., None, :], rotor_offsets_body
    )

    rotor_drag = K_diag * states.rotor_vel[..., None] * rotor_vels_body
    rotor_drag_still = K_diag * states.rotor_vel[..., None] * rotor_vels_body_still
    rotor_drag_summed = jnp.sum(rotor_drag - rotor_drag_still, axis=-2)

    torque_body = jnp.sum(jnp.cross(rotor_offsets_body, rotor_drag - rotor_drag_still), axis=-2)
    torque_world = (rot @ torque_body[..., None])[..., 0]

    force_body = parasitic_drag - parasitic_drag_still + rotor_drag_summed
    force_world = (rot @ force_body[..., None])[..., 0]

    # jdbg.print("wind_world {x}", x=wind_world)
    # jdbg.print("rel_air_world {x}", x=rel_air_world)
    # jdbg.print("v_a_body {x}", x=v_a_body)
    # jdbg.print("force_body {f} torque_body {t}", f=force_body, t=torque_body)
    # jdbg.print("force_world {f} torque_world {t}", f=force_world, t=torque_world)

    return data.replace(states=states.replace(force=force_world, torque=torque_world))


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


def draw_downwash_cone(sim: Sim, world: int = 0, source_drone: int = 0):
    """Draw the far-field downwash cone contour for one source drone."""
    source_pos = np.array(sim.data.states.pos[world, source_drone])
    arm_length = np.array(sim.data.params.L)
    if arm_length.ndim >= 3:
        arm_length = arm_length[world, source_drone, 0]
    motor_distance = 2.0 * float(arm_length)

    rgba = np.array([0.1, 0.55, 1.0, 0.45])
    angles = np.linspace(0.0, 2.0 * np.pi, DOWNWASH_CONE_SEGMENTS + 1)
    unit_circle = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    depths = np.linspace(0.0, DOWNWASH_CONE_DEPTH, DOWNWASH_CONE_RINGS + 1)[1:]

    for depth in depths:
        radius = float(jet_half_width(motor_distance=motor_distance, s=depth))
        ring = np.zeros((DOWNWASH_CONE_SEGMENTS + 1, 3))
        ring[:, :2] = source_pos[:2] + radius * unit_circle
        ring[:, 2] = source_pos[2] - depth
        draw_line(sim, ring, rgba=rgba, start_size=1.5, end_size=1.5)

    ray_angles = np.linspace(0.0, 2.0 * np.pi, DOWNWASH_CONE_RAYS, endpoint=False)
    bottom_radius = float(jet_half_width(motor_distance=motor_distance, s=DOWNWASH_CONE_DEPTH))
    for angle in ray_angles:
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        end = source_pos + bottom_radius * direction
        end[2] -= DOWNWASH_CONE_DEPTH
        draw_line(sim, np.array([source_pos, end]), rgba=rgba, start_size=1.5, end_size=1.5)


def main(plot: bool = False):
    sim = Sim(n_drones=2, control="state", drone_model="cf21B_500")

    sim.step_pipeline = sim.step_pipeline[:3] + (downwash_fn,) + sim.step_pipeline[3:]
    sim.build_step_fn()

    pos = []
    force = []
    torque = []
    sim.reset()
    sim.render()
    for i in range(int(SIM_DURATION * sim.control_freq)):
        sim.state_control(straight_line_control(sim, i / sim.control_freq))
        sim.step(sim.freq // sim.control_freq)
        current_pos = np.array(sim.data.states.pos[0])
        current_force = np.array(sim.data.states.force[0])
        current_torque = np.array(sim.data.states.torque[0])
        pos.append(current_pos)
        force.append(current_force)
        torque.append(current_torque)
        draw_downwash_cone(sim)
        draw_force_arrows(sim)
        sim.render()

    sim.close()
    if plot:
        plot_results(pos, force, torque)


def plot_results(pos: list[NDArray], force: list[NDArray], torque: list[NDArray]):
    import matplotlib.pyplot as plt  # noqa: F401

    pos = np.array(pos)
    force = np.array(force)
    torque = np.array(torque)
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

    fig_force, ax_force = plt.subplots(3, 2, sharex=True, figsize=(11, 6))
    for i, label in enumerate(labels):
        ax_force[i, 0].plot(t, force[:, 0, i], label=f"drone 0 force {label}")
        ax_force[i, 0].plot(t, force[:, 1, i], label=f"drone 1 force {label}", linestyle="--")
        ax_force[i, 0].set_ylabel(f"F{label} [N]")
        ax_force[i, 0].legend()

        ax_force[i, 1].plot(t, torque[:, 0, i], label=f"drone 0 torque {label}")
        ax_force[i, 1].plot(t, torque[:, 1, i], label=f"drone 1 torque {label}", linestyle="--")
        ax_force[i, 1].set_ylabel(f"T{label} [Nm]")
        ax_force[i, 1].legend()

    fig_force.suptitle("Downwash disturbance force and torque")
    ax_force[-1, 0].set_xlabel("Time (s)")
    ax_force[-1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(plot=True)
