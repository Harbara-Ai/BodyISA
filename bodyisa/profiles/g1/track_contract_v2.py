"""State-only JOINT_TRAJECTORY_V2 observer shared by offline and online scoring."""
from collections import deque
import math
import numpy as np


class JointTrajectoryObserver:
    def __init__(self, reference, config):
        self.reference=reference
        self.config=config
        self.dt=float(config['control_dt_s'])
        self.frames=int(config['reference_frames'])
        self.times=np.arange(self.frames)*self.dt
        self.joints=np.asarray(reference.joints[:self.frames],float)
        if self.joints.shape!=(self.frames,29):raise ValueError('Reference joint shape')
        if not np.isfinite(self.joints).all():raise ValueError('Nonfinite reference joints')
        motion=np.quantile(self.joints,.95,axis=0)-np.quantile(self.joints,.05,axis=0)
        self.active=np.flatnonzero(motion>=config['active_joint_range_rad'])
        if not len(self.active):raise ValueError('No active reference joints')
        self.errors=[]
        self.active_errors=[]
        self.max_abs_errors=[]
        self.rolling=deque(maxlen=config['rolling_frames'])
        self.consecutive=0
        self.max_consecutive=0
        self.max_rolling=0.0
        self.failure=None
        self.data_error=None
        self.finalized=False

    def _fail(self,code,idx,observed=None,limit=None):
        if self.failure is None:
            self.failure=dict(primary_failure_class=code,
                failed_predicate=code,first_violation_time_s=idx*self.dt,
                reference_time_s=idx*self.dt,observed_value=observed,
                configured_limit=limit,violation_duration_s=self.consecutive*self.dt)

    def observe(self,snapshot,t):
        if self.finalized:raise RuntimeError('Observer finalized')
        idx=len(self.errors)
        if idx>=self.frames:
            self.data_error='EXTRA_FRAME'
            self._fail('REFERENCE_COVERAGE_ERROR',idx,t,(self.frames-1)*self.dt)
            return self.assessment()
        expected=idx*self.dt
        if not math.isfinite(t) or abs(t-expected)>self.config['clock_tolerance_s']:
            self.data_error=f'CLOCK_MISMATCH expected={expected} actual={t}'
            self._fail('REFERENCE_COVERAGE_ERROR',idx,t,expected)
            return self.assessment()
        q=np.asarray(snapshot.joint_position,float)
        if q.shape!=(29,) or not np.isfinite(q).all():
            self.data_error='JOINT_OBSERVATION_UNAVAILABLE'
            self._fail('OBSERVATION_UNAVAILABLE',idx)
            return self.assessment()
        # Post-step q at t is scored against reference q(t). t=0 is the true entry.
        delta=q-self.joints[idx]
        frame=float(np.sqrt(np.mean(delta**2)))
        active=float(np.sqrt(np.mean(delta[self.active]**2)))
        self.errors.append(frame)
        self.active_errors.append(active)
        self.max_abs_errors.append(float(np.max(np.abs(delta))))
        self.rolling.append(frame)
        if frame>self.config['local_frame_threshold_rad']:
            self.consecutive+=1
        else:
            self.consecutive=0
        self.max_consecutive=max(self.max_consecutive,self.consecutive)
        duration=self.consecutive*self.dt
        if duration>self.config['max_consecutive_excess_s']+1e-9:
            self._fail('TRACK_CONTRACT_VIOLATION_LOCAL_DURATION',idx,duration,
                       self.config['max_consecutive_excess_s'])
        if len(self.rolling)==self.config['rolling_frames']:
            mean=float(np.mean(self.rolling))
            self.max_rolling=max(self.max_rolling,mean)
            if mean>self.config['max_rolling_mean_rad']:
                self._fail('TRACK_CONTRACT_VIOLATION_ROLLING',idx,mean,
                           self.config['max_rolling_mean_rad'])
        return self.assessment()

    def finalize(self):
        self.finalized=True
        if len(self.errors)!=self.frames:
            self.data_error=f'INCOMPLETE_COVERAGE {len(self.errors)}/{self.frames}'
            self._fail('REFERENCE_COVERAGE_ERROR',len(self.errors),len(self.errors),self.frames)
        if self.data_error:return self.assessment()
        mean=float(np.mean(self.errors))
        group_mean=float(np.mean(self.active_errors))
        if mean>self.config['mean_frame_rmse_limit_rad']:
            self._fail('TRACK_CONTRACT_VIOLATION_GLOBAL',self.frames-1,mean,
                       self.config['mean_frame_rmse_limit_rad'])
        if group_mean>self.config['active_group_mean_limit_rad']:
            self._fail('TRACK_CONTRACT_VIOLATION_ACTIVE_GROUP',self.frames-1,group_mean,
                       self.config['active_group_mean_limit_rad'])
        return self.assessment()

    def assessment(self):
        metrics=dict(samples=len(self.errors),expected_samples=self.frames,
            mean_of_frame_joint_rmse_rad=float(np.mean(self.errors)) if self.errors else None,
            active_group_mean_of_frame_rmse_rad=float(np.mean(self.active_errors)) if self.active_errors else None,
            max_25frame_mean_rad=self.max_rolling if len(self.errors)>=self.config['rolling_frames'] else None,
            rolling_window=dict(value=self.max_rolling if len(self.errors)>=self.config['rolling_frames'] else None,
                available=len(self.errors)>=self.config['rolling_frames'],
                reason=None if len(self.errors)>=self.config['rolling_frames'] else 'INSUFFICIENT_SAMPLES'),
            max_consecutive_excess_s=self.max_consecutive*self.dt,
            max_abs_single_joint_error_rad=max(self.max_abs_errors,default=None),
            active_joint_indices=self.active.tolist())
        status='NOT_EVALUABLE' if self.data_error else 'FAIL' if self.failure else 'PASS' if self.finalized else 'PENDING'
        return dict(status=status,failure=self.failure,data_error=self.data_error,metrics=metrics)
