from typing import Optional

from pydantic import BaseModel

from .resource import Resource


class Task(BaseModel):
    id: int
    duration: int
    priority: int
    resources: list[Resource]
    resource_count: int = 1
    predecessors: Optional[list["Task"]] = []
    predecessor_delay: int = 0

    def get_resources(self):
        return self.resources
