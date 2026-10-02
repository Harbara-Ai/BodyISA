from .instruction import Instruction, BodyObjective
from .state import ContactSnapshot, StateSnapshot, RobotState
from .outcome import ExecutionOutcome
from .contracts import ExecutionContract, TransitionContract
from .capability import ValidationEnvelope, PoseRegistry, CapabilityRegistry

def __getattr__(name):
    if name == 'ExecutionRuntime':
        from bodyisa.runtime.session import ExecutionRuntime
        return ExecutionRuntime
    if name in ('LocalEvaluationPermit', 'issue_local_evaluation_permit'):
        from bodyisa.runtime import permits
        return getattr(permits, name)
    raise AttributeError(name)
