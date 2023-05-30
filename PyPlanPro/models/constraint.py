from abc import ABC, abstractmethod
from .resource_group import ResourceGroup
from .resource import Resource
from pydantic import BaseModel

class Constraint(BaseModel, ABC):
    @abstractmethod
    def get_resources(self):
        pass

class ResourceGroupConstraint(Constraint):
    resource_group: ResourceGroup
    resource_count: int = 1
    
    def get_resources(self):
        return self.resource_group.resources

class ResourceConstraint(Constraint):
    resource: Resource

    def get_resources(self):
        return [self.resource]