from dataclasses import dataclass, field

@dataclass(frozen=True)
class Instruction:
    opcode: str
    target_id: str | None = None
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BodyObjective:
    semantic: str
    selected_state: str
    target: dict


