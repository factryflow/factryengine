from pydantic import BaseModel, Field, validator

from .resource import Resource, Team


class Assignment(BaseModel):
    entities: list[Resource | Team] = Field(..., min_items=1)
    entity_count: int = Field(1, gt=0)

    def get_resource_ids(self) -> list[tuple[int]]:
        """returns a list of tuples of resource ids for each entity in the assignment"""
        resource_ids = []
        for entity in self.entities:
            if isinstance(entity, Team):
                team_resource_ids = [resource.id for resource in entity.resources]
                resource_ids.append(tuple(team_resource_ids))
            else:
                resource_ids.append(tuple([entity.id]))
        return resource_ids

    def get_unique_resources(self) -> set[Resource]:
        """returns a set of all unique resources required for the assignment"""
        unique_resources = set()
        for entity in self.entities:
            if isinstance(entity, Team):
                unique_resources.update(entity.resources)
            else:
                unique_resources.add(entity)
        return unique_resources


class Task(BaseModel):
    id: int
    name: str = ""
    duration: int = Field(gt=0)
    priority: int = Field(gt=0)
    assignments: list[Assignment] = []
    predecessor_ids: list[int] = []
    predecessor_delay: int = Field(0, gt=0)
    quantity: int = Field(None, gt=0)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.id == other.id
        return False

    def get_unique_resources(self) -> set[Resource]:
        """returns a set of all unique resources required for the task"""
        return set(
            [
                resource
                for assignment in self.assignments
                for resource in assignment.get_unique_resources()
            ]
        )

    @validator("name", pre=True, always=True)
    def set_name(cls, v, values) -> str:
        if v == "":
            return str(values.get("id"))
        return v

    # def get_resource_group_count(self):
    #     """returns the number of resource groups required for the task"""
    #     return len(self.resources)

    # def get_resource_group_indices(self) -> list[list[int]]:
    #     """
    #     returns a list of lists of indices of resources in each resource group
    #     """
    #     counter = count()
    #     return [[next(counter) for _ in sublist] for sublist in self.resources]

    def get_id(self) -> int:
        """returns the task id"""
        return self.id
