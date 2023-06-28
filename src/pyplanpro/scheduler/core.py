from .heuristic_solver.main import HeuristicSolver
from .scheduler_result import SchedulerResult


class Scheduler:
    def __init__(self, tasks):
        self.tasks = tasks
        self.resources = set(
            [resource for task in tasks for resource in task.get_resources()]
        )
        for resource in self.resources:
            resource.merge_intervals()

    def schedule(self):
        heuristic_solver = HeuristicSolver(self.tasks, self.resources)
        result = heuristic_solver.solve()

        scheduler_result = SchedulerResult(result)
        print(scheduler_result.summary())
        return scheduler_result
        return scheduler_result
