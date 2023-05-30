from typing import List, Optional
from pydantic import BaseModel
from .constraint import Constraint

class Task(BaseModel):
    id: int
    duration: int
    priority: int
    constraints: List[Constraint]
    predecessors: Optional[List['Task']] = []
    predecessor_delay: int = 0
    
    def get_resources(self):
        return {resource for constraint in self.constraints for resource in constraint.get_resources()}