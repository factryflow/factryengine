from typing import List, Optional
from pydantic import BaseModel
from .resource_group import ResourceGroup

class Task(BaseModel):
    id: int
    duration: int
    priority: int
    resource_group: ResourceGroup
    predecessors: Optional[List['Task']] = []
    predecessor_delay: int = 0

    # class Config:
    #     # This is needed to allow comparison using id
    #     # Pydantic models are not hashable out of the box
    #     allow_mutation = False

    # def __eq__(self, other):
    #     if isinstance(other, Task):
    #         return self.id == other.id
    #     return NotImplemented

    # def __hash__(self):
    #     return hash(self.id)
    
    def get_resources(self):
        return self.resource_group.resources