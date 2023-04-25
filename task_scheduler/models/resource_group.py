from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .resource import Resource

@dataclass
class ResourceGroup:
    resource_group_id: int
    resources: List[Resource]

    def __eq__(self, other):
        if isinstance(other, ResourceGroup):
            return self.resource_group_id == other.resource_group_id
        return NotImplemented
    
    def __hash__(self):
        return hash(self.resource_group_id)