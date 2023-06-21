from typing import Optional

from pydantic import BaseModel

from .resource import Resource


class Task(BaseModel):
    id: int
    duration: int
    priority: int
    # resource_constraints: list[Union[ResourceGroupConstraint, SingleResourceConstraint]]
    resources: list[Resource]
    resource_count: int = 1
    predecessors: Optional[list["Task"]] = []
    predecessor_delay: int = 0

    def get_resources(self):
        return self.resources

    # def get_resource_options(self):
    #     return [
    #         resources
    #         for constraint in self.resource_constraints
    #         for resources in constraint.get_possible_resources()
    #     ]

    # def get_unique_resource_combinations(self):
    #     all_combinations = set(
    #         frozenset(combination)
    #         for combination in product(*self.get_resource_options())
    #     )
    #     unique_combinations = [
    #         set(s)
    #         for s in all_combinations
    #         if len(s) == len(self.get_resource_options())
    #     ]
    #     return unique_combinations
