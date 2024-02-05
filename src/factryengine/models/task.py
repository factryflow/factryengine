from pydantic import BaseModel, Field, model_validator, validator

from .resource import Resource, ResourceGroup


class Assignment(BaseModel):
    """
    Atlest one assignment is required for the entire duration of the task,
    it could be an operator or worker. If multiple resources are assigned they
    minimize the completion time for the task.
    """

    resource_groups: list[ResourceGroup] = Field(..., min_items=1)
    resource_count: int = Field(None, gt=0)
    use_all_resources: bool = Field(
        False, description="will use all resources available"
    )

    @model_validator(mode="after")
    def check_valid_combinations(self):
        combinations_count = sum(
            [bool(self.resource_count), bool(self.use_all_resources)]
        )
        if combinations_count == 0:
            raise ValueError("Either resource_count or use_all_resources must be set")
        if combinations_count > 1:
            raise ValueError("Only define one of resource_count or use_all_resources")
        return self

    @validator("resource_count", always=True)
    def check_resource_availability(cls, value, values):
        resource_groups = values.get("resource_groups")
        if resource_groups:
            total_available_resources = sum(
                len(group.resources) for group in resource_groups
            )
            if value > total_available_resources:
                raise ValueError(
                    "resource_count exceeds the total available resources in resource_groups"
                )
        return value

    def get_resource_ids(self) -> list[tuple[int]]:
        """returns a list of tuples of resource ids for each resource group in the assignment"""
        resource_ids = []
        for resource_group in self.resource_groups:
            resource_ids.append(
                tuple([resource.id for resource in resource_group.resources])
            )
        return resource_ids

    def get_unique_resources(self) -> set[Resource]:
        """returns a set of all unique resources required for the assignment"""
        unique_resources = set()
        for resource_group in self.resource_groups:
            unique_resources.update(resource_group.resources)
        return unique_resources


class Task(BaseModel):
    id: int
    name: str = ""
    duration: int = Field(gt=0)
    priority: int = Field(gt=0)
    assignments: list[Assignment] = []
    constraints: set[Resource] = set()
    predecessor_ids: set[int] = set()
    predecessor_delay: int = Field(0, gt=0)
    quantity: int = Field(None, gt=0)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.id == other.id
        return False

    @model_validator(mode="after")
    def check_assigments_or_constraints_are_set(self):
        if not self.assignments and not self.constraints:
            raise ValueError("Either assignments or constraints must be set")
        return self

    def get_unique_resources(self) -> set[Resource]:
        """returns a set of all unique resources required for the task"""
        unique_resources = set()
        for assignment in self.assignments:
            unique_resources.update(assignment.get_unique_resources())
        unique_resources.update(self.constraints)
        return unique_resources

    @validator("name", pre=True, always=True)
    def set_name(cls, v, values) -> str:
        if v == "":
            return str(values.get("id"))
        return v

    def get_id(self) -> int:
        """returns the task id"""
        return self.id
