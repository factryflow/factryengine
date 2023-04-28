from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Resource:
    id: int
    priority: int
    availability_slots: List[Tuple[int, int]] = field(default_factory=list)

    def __eq__(self, other):
        if isinstance(other, Resource):
            return self.id == other.id
        return NotImplemented
    
    def __hash__(self):
        return hash(self.id)
    
    def __post_init__(self):
        if self.priority is None:
            self.priority = 0
        # Ensure availability slots have the desired format
        for slot in self.availability_slots:
            if not isinstance(slot, tuple) or len(slot) != 2:
                raise ValueError("Invalid availability slot format")
            if not isinstance(slot[0], int) or not isinstance(slot[1], int):
                raise ValueError("Availability slot start and end must be integers")
            if slot[0] >= slot[1]:
                raise ValueError("Availability slot start must be less than end")
        
        
        # # Ensure no slots overlap
        # for i in range(len(self.availability_slots)):
        #     for j in range(i+1, len(self.availability_slots)):
        #         slot1_start, slot1_end = self.availability_slots[i]
        #         slot2_start, slot2_end = self.availability_slots[j]
        #         if slot1_start <= slot2_start < slot1_end or slot2_start <= slot1_start < slot2_end:
        #             raise ValueError("Availability slots cannot overlap")