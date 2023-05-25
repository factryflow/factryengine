from typing import List, Optional
from pydantic import BaseModel
from .resource import Resource

class ResourceGroup(BaseModel):
    id: int
    resources: List[Resource]
    efficiency_multiplier: float = 1

    # class Config:
    #     # This is needed to allow comparison using id
    #     # Pydantic models are not hashable out of the box
    #     allow_mutation = False

    # def __eq__(self, other):
    #     if isinstance(other, ResourceGroup):
    #         return self.id == other.id
    #     return NotImplemented
    
    # def __hash__(self):
    #     return hash(self.id)
    
    def get_resource_by_id(self, id: int) -> Optional[Resource]:
        """
        Returns the `Resource` object with the given ID, or `None` if it doesn't exist.
        """
        for resource in self.resources:
            if resource.id == id:
                return resource
        return None
