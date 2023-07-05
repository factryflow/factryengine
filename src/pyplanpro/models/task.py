from typing import Optional

from pydantic import BaseModel, validator

from .resource import Resource


class Task(BaseModel):
    id: int
    duration: int
    priority: int
    resources: list[list[Resource]]
    resource_count: int = 1
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
