from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Resource:
    id: int
    availability_slots: List[Tuple[int, int]] = field(default_factory=list)

    def __eq__(self, other):
        if isinstance(other, Resource):
            return self.id == other.id
        return NotImplemented
    
    def __hash__(self):
        return hash(self.id)
    
    def __post_init__(self):
        # Ensure availability slots have the desired format
        for slot in self.availability_slots:
            if not isinstance(slot, tuple) or len(slot) != 2:
                raise ValueError("Invalid availability slot format")
            if not isinstance(slot[0], int) or not isinstance(slot[1], int):
                raise ValueError("Availability slot start and end must be integers")
            if slot[0] >= slot[1]:
                raise ValueError("Availability slot start must be less than end")