from abc import ABC, abstractmethod
from .resource_group import ResourceGroup
from .resource import Resource
from pydantic import BaseModel

class ResourceConstraint(BaseModel, ABC):
    @abstractmethod
    def get_resources(self):
        pass

    @abstractmethod
    def get_possible_resources(self):
        pass

class ResourceGroupConstraint(ResourceConstraint):
    resource_group: ResourceGroup
    resource_count: int = 1
    use_all_resources: bool = False
    
    def get_resources(self):
        return self.resource_group.resources
    
    def get_possible_resources(self):
        return [self.get_resources() for _ in range(self.resource_count)]

class SingleResourceConstraint(ResourceConstraint):
    resource: Resource

    def get_resources(self):
        return [self.resource]
    
    def get_possible_resources(self):
        return [[self.resource]]