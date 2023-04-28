from .heuristic_solver import HeuristicSolver

class Scheduler():
    def __init__(self, tasks, resources):
        self.tasks = tasks
        self.resources = resources
    
    def schedule(self):
        solver = HeuristicSolver(self.tasks, self.resources)
        return solver.solve()
