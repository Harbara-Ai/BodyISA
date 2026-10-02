from dataclasses import dataclass

@dataclass(frozen=True)
class ContactSnapshot:
    geom_id: int
    body_id: int
    foot: str | None
    distance_m: float
    normal_force_n: float
    active: bool


@dataclass(frozen=True)
class StateSnapshot:
    sim_time: float
    wall_time: float
    base_position_world: tuple
    base_orientation_world: tuple  # wxyz
    base_heading_world: float
    base_linear_velocity_world: tuple
    base_angular_velocity_world: tuple
    joint_position: tuple
    joint_velocity: tuple
    root_height: float
    tilt: float
    contact_snapshot: tuple
    support_force: float
    foot_force_max: float
    nonfoot_floor_contact: bool
    body_position_world: tuple


@dataclass
class RobotState:
    snapshot: StateSnapshot
    mode: str = 'UNKNOWN'


