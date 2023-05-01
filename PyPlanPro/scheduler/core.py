from .heuristic_solver import HeuristicSolver
from .scheduler_result import SchedulerResult

class Scheduler():
    def __init__(self, tasks, resources):
        self.tasks = tasks
        self.resources = resources
    
    def schedule(self):
        heuristic_solver = HeuristicSolver(self.tasks, self.resources)
        result = heuristic_solver.solve()

        scheduler_result = SchedulerResult(result)
        print(scheduler_result.summary())
        return scheduler_result
