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
            task_resource_windows_dict = (
                self.window_manager.get_task_resource_windows_dict(
                    task_resource_ids, task_earliest_start
                )
            )

            # allocate task
            allocated_resource_windows_dict = self.task_allocator.allocate_task(
                resource_windows_dict=task_resource_windows_dict,
                task_duration=task.duration,
                resource_count=task.resource_count,
            )

            if not allocated_resource_windows_dict:
                unscheduled_tasks.append(task_id)
                continue

            # update resource windows
            self.window_manager.update_resource_windows(allocated_resource_windows_dict)

            # Append task values
            task_values = {
                "task_id": task_id,
                "assigned_resource_ids": list(allocated_resource_windows_dict.keys()),
                "task_start": [
                    start for start, end in allocated_resource_windows_dict.values()
                ],
                "task_end": [
                    end for start, end in allocated_resource_windows_dict.values()
                ],
            }
            self.task_vars[task_id] = task_values

        return list(
            self.task_vars.values()
        )  # Return values of the dictionary as a list

    def _get_task_earliest_start(self, task):
        """
        Retuns the earliest start of a task based on the latest end of its predecessors.
        """
        return max(
            [
                start
                for pred in task.predecessors
                for start in self.task_vars[pred.id]["task_end"]
            ],
            default=0,
        )
