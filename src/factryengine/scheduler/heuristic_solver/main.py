import numpy as np

from ...models import Resource, Task
from ..utils import get_task_predecessors
from .exceptions import AllocationError
from .task_allocator import TaskAllocator
from .window_manager import WindowManager


class HeuristicSolver:
    def __init__(
        self,
        task_dict: dict[str, Task],
        resources: set[Resource],
        task_order: list[str],
    ):
        self.task_dict = task_dict
        self.task_order = task_order
        self.task_allocator = TaskAllocator()
        self.window_manager = WindowManager(resources)
        self.task_vars = {
            task_id: {
                "task_id": task_id,
                "assigned_resource_ids": None,
                "task_start": None,
                "task_end": None,
                "resource_intervals": None,
            }
            for task_id in self.task_dict.keys()
        }

    def solve(self):
        unscheduled_tasks = []

        for task_id in self.task_order:
            task = self.task_dict[task_id]

            task_earliest_start = self._get_task_earliest_start(task, self.task_dict)

            if task_earliest_start is None:
                unscheduled_tasks.append(task_id)
                continue

            # get task resources and windows dict
            task_resource_ids = np.array(
                [resource.id for resource in task.get_unique_resources()]
            )

            task_resource_windows_dict = (
                self.window_manager.get_task_resource_windows_dict(
                    task_resource_ids, task_earliest_start
                )
            )

            if task_resource_windows_dict == {}:
                unscheduled_tasks.append(task_id)
                continue

            # allocate task
            try:
                allocated_resource_windows_dict = self.task_allocator.allocate_task(
                    resource_windows_dict=task_resource_windows_dict,
                    assignments=task.assignments,
                    task_duration=task.duration,
                )
            except AllocationError:
                unscheduled_tasks.append(task_id)
                continue

            # update resource windows
            self.window_manager.update_resource_windows(allocated_resource_windows_dict)

            # Append task values
            task_values = {
                "task_id": task_id,
                "assigned_resource_ids": list(allocated_resource_windows_dict.keys()),
                "task_start": min(
                    start for start, _ in allocated_resource_windows_dict.values()
                ),
                "task_end": max(
                    end for _, end in allocated_resource_windows_dict.values()
                ),
                "resource_intervals": allocated_resource_windows_dict.values(),
            }
            self.task_vars[task_id] = task_values

        return list(
            self.task_vars.values()
        )  # Return values of the dictionary as a list

    def _get_task_earliest_start(self, task: Task, task_dict: dict) -> int | None:
        """
        Retuns the earliest start of a task based on the latest end of its predecessors.
        """
        task_ends = []

        predecessors = get_task_predecessors(task, task_dict)

        for pred in predecessors:
            task_end = self.task_vars[pred.id]["task_end"]
            if task_end is None:
                return None
            task_ends.append(task_end + task.predecessor_delay)

        return max(task_ends, default=0)
