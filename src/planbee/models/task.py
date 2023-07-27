from itertools import count

from pydantic import BaseModel, Field, PrivateAttr, validator

from .resource import Resource


class Task(BaseModel):
    id: int | str
    duration: int = Field(gt=0)
    priority: int = Field(gt=0)
    resources: list[set[Resource]]
    resource_count: int | str = 1
    predecessors: list["Task"] = []
    predecessor_delay: int = Field(0, gt=0)
    batch_size: int = Field(None, gt=0)
    quantity: int = Field(None, gt=0)
    _batch_id: int = PrivateAttr(None)

    @property
    def uid(self) -> str:
        """returns the unique id of the task"""
        if self.batch_id is None:
            return str(self.id)
        else:
            return f"{self.id}-{self.batch_id}"

    @property
    def batch_id(self):
        """returns the batch id of the task"""
        return self._batch_id

    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.uid == other.uid
        return False

    @validator("resources", pre=True)
    def ensure_list(cls, v):
        """ensures that the resources are in the form of a list of lists"""
        if not isinstance(v, list):  # if a single resource object is passed
            return [[v]]  # make it a list of list
        if (
            isinstance(v, list) and len(v) > 0 and not isinstance(v[0], list)
        ):  # if a list of resources is passed
            return [v]  # make it a list of list
        return v  # if a list of lists is passed, return as it is

    @validator("resource_count", always=True)
    def set_resource_count(cls, v, values):
        """
        sets the resource count of the task. If resource_count is set to "all", it is
        set to the maximum number of resources in any resource group
        """
        if isinstance(v, str) and v.lower() == "all":
            if "resources" in values:
                return max(
                    len(resource_group) for resource_group in values["resources"]
                )

        elif isinstance(v, int):
            return v
        else:
            raise ValueError("Invalid value for resource_count.")

    def set_batch_id(self, batch_id):
        """sets the batch id of the task"""
        self._batch_id = batch_id

    def get_resources(self):
        """returns a list of all resources required for the task"""
        return [
            resource for resource_list in self.resources for resource in resource_list
        ]

    def get_resource_group_count(self):
        """returns the number of resource groups required for the task"""
        return len(self.resources)

    def get_resource_group_indices(self) -> list[list[int]]:
        """
        returns a list of lists of indices of resources in each resource group
        """
        counter = count()
        return [[next(counter) for _ in sublist] for sublist in self.resources]

    def is_splittable(self):
        """
        Checks if the task is splittable into batches.
        """
        return (
            self.batch_size is not None
            and self.quantity is not None
            and self.batch_size > 0
            and self.quantity > self.batch_size
        )
