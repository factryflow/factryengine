from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Resource:
    id: int
    availability_slots: List = field(default_factory=list)

    def __eq__(self, other):
        if isinstance(other, Resource):
            return self.id == other.id
        return NotImplemented
    
    def __hash__(self):
        return hash(self.id)