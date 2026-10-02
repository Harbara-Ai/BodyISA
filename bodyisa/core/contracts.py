from dataclasses import dataclass
from collections import deque
import numpy as np

class ExecutionContract:
    def __init__(self, config, mass, target=None):
        self.config, self.mass, self.target = config, mass, target
        self.hold_start = None
        self.heights = deque()

    def evaluate(self, s, elapsed, timeout):
        c = self.config
        numeric = [*s.base_position_world, *s.base_orientation_world,
                   *s.base_linear_velocity_world, *s.base_angular_velocity_world,
                   *s.joint_position, *s.joint_velocity, s.tilt,
                   s.support_force, s.foot_force_max, s.sim_time]
        violations = []
        if not np.isfinite(numeric).all(): violations.append('NONFINITE_STATE')
        if s.tilt > c['fall_tilt']: violations.append('FALL_TILT')
        if s.root_height < c['fall_height']: violations.append('FALL_HEIGHT')
        if s.nonfoot_floor_contact: violations.append('NONFOOT_FLOOR_CONTACT')
        errors = dict(tilt=s.tilt, height_error=abs(s.root_height-c['height']),
            linear_speed=float(np.linalg.norm(s.base_linear_velocity_world)),
            angular_speed=float(np.linalg.norm(s.base_angular_velocity_world)),
            joint_speed=float(np.max(np.abs(s.joint_velocity))))
        predicates = dict(upright=errors['tilt'] <= c['tilt'],
            height=errors['height_error'] <= c['height_tolerance'],
            low_linear_speed=errors['linear_speed'] <= c['linear_speed'],
            low_angular_speed=errors['angular_speed'] <= c['angular_speed'],
            low_joint_speed=errors['joint_speed'] <= c['joint_speed'],
            support=s.foot_force_max >= c['support_foot_force'] and
                    s.support_force >= self.mass*9.81*c['support_weight_fraction'])
        if self.target is not None:
            diff = np.asarray(s.joint_position)-self.target
            errors.update(joint_rmse=float(np.sqrt(np.mean(diff**2))),
                          joint_max_error=float(np.max(np.abs(diff))))
            predicates.update(pose_rmse=errors['joint_rmse'] <= c['pose_rmse'],
                              pose_max_error=errors['joint_max_error'] <= c['pose_max_error'])
        if not all(predicates.values()):
            self.hold_start = None
            self.heights.clear()
        else:
            self.heights.append(s.root_height)
            if max(self.heights)-min(self.heights) > c['height_span']:
                self.hold_start = s.sim_time
                self.heights.clear()
                self.heights.append(s.root_height)
            if self.hold_start is None: self.hold_start = s.sim_time
        hold = 0.0 if self.hold_start is None else s.sim_time-self.hold_start
        predicates['stable_height'] = bool(self.heights) and max(self.heights)-min(self.heights) <= c['height_span']
        errors['hold_elapsed_s'] = hold
        reason = ('INVARIANT_VIOLATION' if violations else 'TIMEOUT' if elapsed >= timeout-1e-9
                  else 'SUCCESS' if all(predicates.values()) and hold >= c['hold_s']-1e-9 else None)
        return predicates, errors, violations, reason


@dataclass(frozen=True)
class TransitionContract:
    source: str
    target: str
    timeout_s: float
    entry: str = 'CURRENT_STAND_REGION'
    abort: str = 'INVARIANT_VIOLATION_OR_TIMEOUT'

    @staticmethod
    def between(source, target, timeout_s):
        allowed = {('STAND','REACH_POSE'), ('REACH_POSE','STAND'),
                   ('STAND','TRACK_MOTION'), ('TRACK_MOTION','STAND')}
        if (source,target) not in allowed: raise ValueError('Unsupported transition')
        return TransitionContract(source,target,timeout_s,
            'CURRENT_STAND_REGION' if source == 'STAND' else 'STABILIZE_AT_DESTINATION')


