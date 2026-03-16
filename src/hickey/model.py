import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, MAXYEAR
from enum import Enum


class MemoryType(Enum):
    CORRECTION    = (1.5, 90)
    DECISION      = (1.2, 60)
    FACT          = (1.0, 30)
    PREFERENCE    = (1.1, 60)
    INVESTIGATION = (0.8, 21)

    def __init__(self, weight: float, halflife: int):
        self.weight = weight
        self.halflife = halflife


@dataclass
class Memory:
    content: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    type: MemoryType = MemoryType.FACT
    project: str = field(default_factory=lambda: os.path.basename(os.getcwd()))
    tags: list[str] = field(default_factory=list)
    auto: bool = False
    confidence: float = 1.0
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires: datetime = field(default_factory=lambda: datetime(MAXYEAR, 12, 31, 23, 59, 59, 999999))
