"""Finite entry neighborhoods and synchronous runtime-owned source receipts."""
from dataclasses import dataclass
import json
import math
import numpy as np
from .evidence import canonical,digest


@dataclass(frozen=True)
class SourceReceipt:
    session_id: str
    episode_id: str
    opcode: str
    contract_hash: str
    outcome: str
    execution_id: str
    terminal_sim_time: float
    context_json: str
    evidence_revision: int

    def context(self):return json.loads(self.context_json)


class EntryPolicy:
    def __init__(self,config):
        self._json=canonical(config)
        self.identity=digest(config)
    @property
    def config(self):return json.loads(self._json)

    def region(self,kind,snapshot):
        region=self.config['regions'][kind]
        center=region['center'];limits=region['limits']
        q=np.asarray(snapshot.joint_position)-np.asarray(center['joint_position'])
        actual=dict(joint_rmse_rad=float(np.sqrt(np.mean(q*q))),joint_max_rad=float(np.max(np.abs(q))),
            joint_speed_rad_s=float(np.max(np.abs(snapshot.joint_velocity))),
            linear_speed_m_s=float(np.linalg.norm(snapshot.base_linear_velocity_world)),
            angular_speed_rad_s=float(np.linalg.norm(snapshot.base_angular_velocity_world)),
            tilt_rad=snapshot.tilt,height_delta_m=abs(snapshot.root_height-center['root_height']),
            root_xy_distance_m=float(np.linalg.norm(np.asarray(snapshot.base_position_world[:2])-np.asarray(center['base_position_world'][:2]))),
            heading_distance_rad=abs(math.remainder(snapshot.base_heading_world-center['base_heading_world'],2*math.pi)))
        predicates={key:math.isfinite(value) and value<=limits[key]+1e-12 for key,value in actual.items()}
        return dict(region=kind,values=actual,limits=limits,predicates=predicates,
            eligible=all(predicates.values()),reason=None if all(predicates.values()) else 'ENTRY_REGION_OUTSIDE')

    def check(self,opcode,snapshot,phase,receipt,context,session_id,episode_id,evidence_revision,initial_context):
        if opcode=='STAND':
            if phase=='INITIALIZED':
                if initial_context!=context:return dict(eligible=False,reason='INITIALIZATION_CONTEXT_CHANGED')
                return self.region('INITIALIZED_NOMINAL',snapshot)
            if phase=='TRACK_DONE':kind='TRACK_EXIT_STABLE';source='TRACK_MOTION'
            else:return dict(eligible=False,reason='UNSUPPORTED_STAND_CONTEXT')
        elif opcode=='TRACK_MOTION':
            if phase!='STAND_DONE':return dict(eligible=False,reason='REQUIRES_STAND' if phase in ('INITIALIZED','UNINITIALIZED') else 'UNSUPPORTED_TRACK_CONTEXT')
            kind='STAND_TERMINAL';source='STAND'
        else:return dict(eligible=False,reason='ENTRY_POLICY_UNSUPPORTED_OPCODE')
        if receipt is None or receipt.opcode!=source or receipt.outcome!='SUCCESS':
            return dict(eligible=False,reason='SOURCE_RECEIPT_MISSING')
        if receipt.session_id!=session_id or receipt.episode_id!=episode_id or receipt.evidence_revision!=evidence_revision:
            return dict(eligible=False,reason='SOURCE_RECEIPT_CONTEXT_REVISION')
        if receipt.context()!=context:return dict(eligible=False,reason='SOURCE_STATE_OR_HISTORY_CHANGED')
        return self.region(kind,snapshot)
