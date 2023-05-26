from typing import List, Tuple
from pydantic import BaseModel, validator

class Resource(BaseModel):
    id: int
    availability_slots: List[Tuple[int, int]] = []
    efficiency_multiplier: float = 1

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Resource):
            return self.id == other.id
        return False

    @validator('availability_slots', each_item=True)
    def validate_slots(cls, slot):
        if slot[0] >= slot[1]:
            raise ValueError("Availability slot start must be less than end")
        return slot
