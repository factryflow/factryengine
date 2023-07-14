from ..models import Task
from .heuristic_solver.main import HeuristicSolver
from .scheduler_result import SchedulerResult
from .task_graph import TaskGraph


class Scheduler:
    def __init__(self, tasks: list[Task]):
        self.tasks = tasks
        self.resources = set(
            [resource for task in tasks for resource in task.get_resources()]
        )
        self.task_graph = TaskGraph(tasks)

    def schedule(self):
        # merge intervals for all resources
        for resource in self.resources:
            resource.merge_intervals()

        # schedule tasks
        task_order = self.task_graph.get_task_order()
        heuristic_solver = HeuristicSolver(self.tasks, self.resources, task_order)
        result = heuristic_solver.solve()

        scheduler_result = SchedulerResult(result)
        print(scheduler_result.summary())
        return scheduler_result
