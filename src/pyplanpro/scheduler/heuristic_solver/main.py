from .task_allocator import TaskAllocator
from .task_graph import TaskGraph
from .window_manager import WindowManager


class HeuristicSolver:
    def __init__(self, tasks, resources):
        self.task_allocator = TaskAllocator()
        self.task_graph = TaskGraph(tasks)
        self.window_manager = WindowManager(resources)
        self.resource_windows_dict = WindowManager.create_resource_windows_dict()
        self.task_vars = {
            task.id: {
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
            resources = task.get_resources()
            resource_ids = [resource.id for resource in resources]
            resource_windows_dict = {
                id: self.resource_windows_dict[id] for id in resource_ids
            }

            allocated_resource_windows_dict = (
                self.task_allocator.find_earliest_solution(
                    task.duration, resource_windows_dict, task.resource_count
                )
            )
            if allocated_resource_windows_dict is None:
                unscheduled_tasks.append(task_id)
                continue

            # update resource windows
            for (
                resource_id,
                allocated_window,
            ) in allocated_resource_windows_dict.items():
                window = self.resource_windows_dict[resource_id]
                window_trimmed = WindowManager.trim_windows(window, window_trimmed)
                self.resource_windows_dict[resource_id] = window_trimmed

            # Append task values
            task_values = {
                "task_id": task_id,
                "assigned_resource_id": allocated_resource_windows_dict.keys(),
                "task_start": allocated_resource_windows_dict[resource_ids[0]][0],
                "task_end": allocated_resource_windows_dict[resource_ids[0]][1],
            }
            self.task_vars[task_id] = task_values

        return list(
            self.task_vars.values()
        )  # Return values of the dictionary as a list
