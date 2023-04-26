from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .resource import Resource

@dataclass
class ResourceGroup:
    id: int
    resources: List[Resource]

    def __eq__(self, other):
        if isinstance(other, ResourceGroup):
            return self.id == other.id
        return NotImplemented
    
    def __hash__(self):
        return hash(self.id)
    
    def get_resource_by_id(self, id: int) -> Optional[Resource]:
        """
        Returns the `Resource` object with the given ID, or `None` if it doesn't exist.
        """
        for resource in self.resources:
            if resource.id == id:
                return resource
        return None