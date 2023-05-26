from .heuristic_solver import HeuristicSolver
from .scheduler_result import SchedulerResult

class Scheduler():
    def __init__(self, tasks):
        self.tasks = tasks
        self.resource_groups = set([task.resource_group for task in tasks])
        self.resources = set([resource for rg in self.resource_groups for resource in rg.resources])
    
    def schedule(self):
        heuristic_solver = HeuristicSolver(self.tasks, self.resources)
        result = heuristic_solver.solve()

        scheduler_result = SchedulerResult(result)
        print(scheduler_result.summary())
        return scheduler_result
