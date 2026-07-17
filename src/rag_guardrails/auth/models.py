from dataclasses import dataclass, field
from enum import IntEnum


class ClearanceLevel(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3

    @classmethod
    def from_str(cls, name: str) -> "ClearanceLevel":
        try:
            return cls[name.strip().upper()]
        except KeyError:
            raise ValueError(
                f"Unknown clearance level {name!r}; expected one of "
                f"{[m.name.lower() for m in cls]}"
            ) from None


@dataclass
class UserContext:
    user_id: str
    clearance: ClearanceLevel
    roles: list[str] = field(default_factory=list)
