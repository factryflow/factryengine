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
    
    def get_resources(self):
        return self.resource_group.resources