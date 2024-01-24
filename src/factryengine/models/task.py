from pydantic import BaseModel, Field, validator

from .resource import Resource, Team


class Assignment(BaseModel):
    entities: list[Resource | Team] = Field(..., min_items=1)
    entity_count: int = Field(1, gt=0)

    def get_resource_groups(self) -> list[tuple[Resource]]:
        """returns a list of all resources required for the assignment"""
        resources = []
        teams_resources = []
        for entity in self.entities:
            if isinstance(entity, Resource):
                resources.append(entity)
            else:
                teams_resources.append(tuple(entity.resources))
        return [tuple(resources)] + teams_resources


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
                for groups in assignment.get_resource_groups()
                for resource in groups
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
