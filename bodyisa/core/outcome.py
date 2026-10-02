from dataclasses import dataclass
from .instruction import Instruction
from .state import StateSnapshot

@dataclass(frozen=True)
class ExecutionOutcome:
    instruction: Instruction
    status: str
    termination_reason: str
    elapsed_sim_s: float
    initial: StateSnapshot
    terminal: StateSnapshot
    evidence_path: str
    semantic_assessment: dict | None = None
    exit_stabilization_outcome: str | None = None
    primary_failure_class: str | None = None


