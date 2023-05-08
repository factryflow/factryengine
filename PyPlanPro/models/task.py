from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .resource_group import ResourceGroup

@dataclass
class Task:
    id: int
    duration: int
    priority: int
    resource_group: ResourceGroup
    predecessors: Optional[List['Task']] = field(default_factory=list)
    predecessor_delay: int = 0

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.id == other.id
        return NotImplemented

    def __hash__(self):
        return hash(self.id)
    
    def get_resources(self):
        return self.resource_group.resources