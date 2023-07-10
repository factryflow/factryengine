import numpy as np

from .task_allocator import TaskAllocator
from .task_graph import TaskGraph
from .window_manager import WindowManager


class HeuristicSolver:
    def __init__(self, tasks, resources):
        self.task_allocator = TaskAllocator()
        self.task_graph = TaskGraph(tasks)
        self.window_manager = WindowManager(resources)
        self.task_vars = {
            task.id: {
                "task_id": task.id,
                "assigned_resource_ids": None,
                "task_start": None,
                "task_end": None,
                "resource_intervals": None,
            }
            for task in tasks
        }
        self.task_dict = {task.id: task for task in tasks}

    def solve(self):
        task_order = self.task_graph.get_task_order()
        unscheduled_tasks = []

        for task_id in task_order:
            task = self.task_dict[task_id]

            # get task resources and windows dict
            task_resource_ids = [resource.id for resource in task.get_resources()]
            task_earliest_start = self._get_task_earliest_start(task)

            if task_earliest_start is None:
                unscheduled_tasks.append(task_id)
                continue

            task_resource_windows_dict = (
                self.window_manager.get_task_resource_windows_dict(
                    task_resource_ids, task_earliest_start
                )
            )
            if not task_resource_windows_dict:
                unscheduled_tasks.append(task_id)
                continue

            # allocate task
            allocated_resource_windows_dict = self.task_allocator.allocate_task(
                resource_windows_dict=task_resource_windows_dict,
                task_duration=task.duration,
                resource_count=task.resource_count,
                resource_group_indices=task.get_resource_group_indices(),
            )

            if not allocated_resource_windows_dict:
                unscheduled_tasks.append(task_id)
                continue

            resource_windows_min_max = self.min_max_dict_np(
                allocated_resource_windows_dict
            )

            # update resource windows
            self.window_manager.update_resource_windows(resource_windows_min_max)

            # Append task values
            task_values = {
                "task_id": task_id,
                "assigned_resource_ids": list(allocated_resource_windows_dict.keys()),
                "task_start": min(
                    start for start, _ in resource_windows_min_max.values()
                ),
                "task_end": max(end for _, end in resource_windows_min_max.values()),
                "resource_intervals": allocated_resource_windows_dict.values(),
            }
            self.task_vars[task_id] = task_values

        return list(
            self.task_vars.values()
        )  # Return values of the dictionary as a list

    def _get_task_earliest_start(self, task):
        """
        Retuns the earliest start of a task based on the latest end of its predecessors.
        """
        task_ends = []
        for pred in task.predecessors:
            task_end = self.task_vars[pred.id]["task_end"]
            if task_end is None:
                return None
            task_ends.append(task_end + task.predecessor_delay)

        return max(task_ends, default=0)

    def min_max_dict_np(self, d):
        result = {}

        for key, value_list in d.items():
            min_val = np.min([x[0] for x in value_list])
            max_val = np.max([x[1] for x in value_list])
            result[key] = (min_val, max_val)

        return result
        return result
