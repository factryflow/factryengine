from itertools import count
from typing import Optional, Union

from pydantic import BaseModel, validator

from .resource import Resource


class Task(BaseModel):
    id: int
    duration: int
    priority: int
    resources: list[list[Resource]]
    resource_count: Union[int, str] = 1
    predecessors: Optional[list["Task"]] = []
    predecessor_delay: int = 0

    @validator("resources", pre=True)
    def ensure_list(cls, v):
        if not isinstance(v, list):  # if a single resource object is passed
            return [[v]]  # make it a list of list
        if (
            isinstance(v, list) and len(v) > 0 and not isinstance(v[0], list)
        ):  # if a list of resources is passed
            return [v]  # make it a list of list
        return v  # if a list of lists is passed, return as it is

    def get_resources(self):
        return [
            resource for resource_list in self.resources for resource in resource_list
        ]

    def get_resource_group_count(self):
        return len(self.resources)

    def get_resource_group_indices(self) -> list[list[int]]:
        """
        returns a list of lists of indices of resources in each resource group
        """
        counter = count()
        return [[next(counter) for _ in sublist] for sublist in self.resources]

    @validator("resource_count", always=True)
    def set_resource_count(cls, v, values):
        if isinstance(v, str) and v.lower() == "all":
            if "resources" in values:
                return max(
                    len(resource_group) for resource_group in values["resources"]
                )

        elif isinstance(v, int):
            return v
        else:
            raise ValueError("Invalid value for resource_count.")
