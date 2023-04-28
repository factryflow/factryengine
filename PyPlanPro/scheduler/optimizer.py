from ortools.sat.python import cp_model
import time

class Optimizer:

    def __init__(
            self, 
            tasks, 
            horizon,
            time_limit=None,
            solution_limit=None,
            num_search_workers=None, 
            optimality_tolerance=None
            ):
        
        self.tasks = tasks
        self.horizon = horizon
        self.resource_bools = None
        self.subtask_vars = None
        self.time_limit = time_limit
        self.solution_limit = solution_limit
        self.num_search_workers = num_search_workers
        self.optimality_tolerance = optimality_tolerance

    def schedule(self):
        preprocessing_time = time.time()
        # get data
        tasks = self.tasks
        horizon = self.horizon

        resource_groups = set(task.resource_group for task in tasks)
        resources = set(resource for resource_group in resource_groups for resource in resource_group.resources) 
        self.resource_groups = resource_groups
        self.resources = resources
        
        model = cp_model.CpModel()

        # create resource bools
        resource_bools = {}
        for task in tasks:
            for resource in task.get_resources():
                resource_bools[task.id, resource.id] = model.NewBoolVar(f'x[{task.id},{resource.id}]')
        
        self.resource_bools = resource_bools

        # constrain tasks to one resource
        for task in tasks:
            model.AddExactlyOne([resource_bools[task.id, resource.id] for resource in task.get_resources()])

        # subtaks 
        subtask_vars = {(task.id,resource.id): [] for task in tasks for resource in task.get_resources()}
        task_vars = {task.id: {} for task in tasks}
        # Create subtasks for each availability slot
        for task in tasks:
            task_duration = task.duration

            # set task_start and end variables
            task_start = model.NewIntVar(0, horizon, f'task_start_{task.id}')
            task_end = model.NewIntVar(0, horizon, f'task_end_{task.id}')
            task_vars[task.id]["start"] = task_start
            task_vars[task.id]["end"] = task_end
            for resource in task.get_resources():
                subtask_durations = []
                for subtask_id, (slot_start, slot_end) in enumerate(resource.availability_slots):

                    # create variables
                    start = model.NewIntVar(slot_start, slot_end, f'start_{slot_start}')
                    end = model.NewIntVar(slot_start, slot_end, f'end_{slot_end}')
                    duration = model.NewIntVar(0, slot_end-slot_start, f'duration_{subtask_id}')
                    subtask_durations.append(duration)
                    interval = model.NewIntervalVar(start, duration, end, f'interval_{task.id, subtask_id}')

                    # Create a BoolVar if duration greater than 0
                    duration_gt_zero = model.NewBoolVar(f'duration_gt_zero_{task.id}_{subtask_id}')
                    model.Add(duration > 0).OnlyEnforceIf(duration_gt_zero)
                    model.Add(duration == 0).OnlyEnforceIf(duration_gt_zero.Not())

                    # Craete a BoolVar to check if duration equal sum of subtask durations
                    duration_eq_sum_duration = model.NewBoolVar(f'duration_eq_sum_duration_{task.id}_{subtask_id}')
                    model.Add(sum(subtask_durations) == duration).OnlyEnforceIf(duration_eq_sum_duration)
                    model.Add(sum(subtask_durations) != duration).OnlyEnforceIf(duration_eq_sum_duration.Not())
    
                    # Craete a BoolVar to check if tasks has ended
                    has_ended = model.NewBoolVar(f'has_ended{task.id}_{subtask_id}')
                    model.Add(sum(subtask_durations) >= task_duration).OnlyEnforceIf(has_ended)
                    model.Add(sum(subtask_durations) < task_duration).OnlyEnforceIf(has_ended.Not())

                    # Create a BoolVar to check if slot is task start
                    is_task_start = model.NewBoolVar(f'is_task_start_{task.id}_{subtask_id}')
                    model.AddBoolAnd([duration_gt_zero, duration_eq_sum_duration]).OnlyEnforceIf(is_task_start)
                    model.AddBoolOr([duration_gt_zero.Not(), duration_eq_sum_duration.Not()]).OnlyEnforceIf(is_task_start.Not())

                    # Create a BoolVar to check if task has ended
                    is_task_end = model.NewBoolVar(f'is_task_end_{task.id}_{subtask_id}')
                    model.AddBoolAnd([has_ended, duration_gt_zero]).OnlyEnforceIf(is_task_end)
                    model.AddBoolOr([has_ended.Not(), duration_gt_zero.Not()]).OnlyEnforceIf(is_task_end.Not())

                    # Ensure is_in_progress is true when subtask_durations is between > 0 and 10
                    is_in_progress = model.NewBoolVar(f'task_started_{task.id}_{subtask_id}')
                    model.Add(sum(subtask_durations) > 0).OnlyEnforceIf(is_in_progress)
                    model.Add(sum(subtask_durations) == 0).OnlyEnforceIf(is_in_progress.Not())
                    
                    # Ensure subtasks are continuous 
                    # if task is in progress it should fill the whole slot
                    model.Add(start == slot_start).OnlyEnforceIf((is_in_progress, has_ended.Not(), is_task_start.Not()))
                    model.Add(end == slot_end).OnlyEnforceIf((is_in_progress, has_ended.Not(), is_task_start.Not()))

                    # if task is starting but not ending
                    model.Add(end == slot_end).OnlyEnforceIf((is_task_start, is_task_end.Not()))

                    # if task is ending but not starting
                    model.Add(start == slot_start).OnlyEnforceIf((is_task_end,is_task_start.Not()))        

                    # set task_stat and task_end
                    model.Add(task_start == start).OnlyEnforceIf(is_task_start).OnlyEnforceIf(resource_bools[task.id,resource.id])       
                    model.Add(task_end == end).OnlyEnforceIf(is_task_end).OnlyEnforceIf(resource_bools[task.id,resource.id])

                    # Add the subtask to the list
                    subtask_vars[task.id,resource.id].append({
                        "subtask_id" : subtask_id,
                        "slot_id": subtask_id,
                        "start" : start,
                        "duration" : duration,
                        "end" : end,
                        "interval" : interval,
                        "is_task_start" : is_task_start,
                        "is_in_progress" : is_in_progress,
                        "is_task_end" : is_task_end,
                        "duration_gt_zero" : duration_gt_zero
                    })

                # Add constraint to enforce the sum of durations of subtasks to be equal to task_duration
                model.Add(sum(subtask_durations) == task_duration).OnlyEnforceIf(resource_bools[task.id,resource.id])
                model.Add(sum(subtask_durations) == 0).OnlyEnforceIf(resource_bools[task.id,resource.id].Not())

        self.subtask_vars = subtask_vars

        # Add precedence constraint
        for task in tasks:
            for predecessor in task.predecessors:
                model.Add(task_vars[task.id]["start"] >= task_vars[predecessor.id]["end"])

        # Add NoOverlap constraint for each resource.
        resource_intervals = {resource.id: [] for resource in resources}
        for (task_id, resource_id), subtask_list in subtask_vars.items():
            for subtask in subtask_list:
                optional_interval = model.NewOptionalIntervalVar(
                    start = subtask["start"], 
                    size = subtask["duration"], 
                    end = subtask["end"], 
                    is_present = resource_bools[task_id, resource_id],
                    name = f'resource_interval_{resource}_{task_id, subtask["subtask_id"]}'
                    )
                resource_intervals[resource_id].append(optional_interval)

        for resource in resources:
            model.AddNoOverlap(resource_intervals[resource.id])

        # object var to reduce makespan
        # only consider ends with duration > 0
        optional_ends = []
        for (task_id, resource_id), subtask_list in subtask_vars.items():
            for subtask in subtask_list:
                end_var = model.NewIntVar(0, horizon, f'end_var_{task_id}_{resource_id}_{subtask["subtask_id"]}')
                model.AddElement(subtask["duration_gt_zero"], [0,subtask["end"]], end_var)
                optional_ends.append(end_var)

        obj_var = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(obj_var, optional_ends)
        model.Minimize(obj_var)
        
        # Create a solver and solve the model
        solver = cp_model.CpSolver()

        solution_printer = self.VarArraySolutionPrinterWithLimit([obj_var], self.solution_limit)

        # Set solver parameters if provided
        if self.time_limit is not None:
            solver.parameters.max_time_in_seconds = self.time_limit

        if self.num_search_workers is not None:
            solver.parameters.num_search_workers = self.num_search_workers

        if self.optimality_tolerance is not None:
            solver.parameters.relative_gap_limit = self.optimality_tolerance

        # Enumerate all solutions.
        solver.parameters.enumerate_all_solutions = True

        solver_start_time = time.time()
        print(f"[INFO] Setup time: {round((solver_start_time- preprocessing_time),2)} seconds")

        status = solver.Solve(model, solution_printer)
        solver_end_time = time.time()

        print(f'[INFO] Solver time: {round((solver_end_time - solver_start_time),2)} seconds')
        # Check the solver status and print the solution
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"\nMakespan = {solver.ObjectiveValue()}")
            for task in tasks:
                for resource in task.get_resources():
                    if solver.BooleanValue(resource_bools[task.id, resource.id]):
                        print(
                            f"\n[Task {task.id}] - Assigned Resource: {resource.id} - "
                            f"Span: {solver.Value(task_vars[task.id]['start'])} to "
                            f"{solver.Value(task_vars[task.id]['end'])}"
                            )
                        for subtask in subtask_vars[task.id, resource.id]:
                            # if solver.Value(subtask['duration']) > 0:
                                print(
                                    f"- Subtask {subtask['subtask_id']}: "
                                    f"start: {solver.Value(subtask['start'])}, "
                                    f"end: {solver.Value(subtask['end'])}, "
                                    f"duration: {solver.Value(subtask['duration'])}, "
                                    f"bools: {[solver.Value(subtask[field]) for field in ['is_task_start', 'is_in_progress', 'is_task_end', 'duration_gt_zero']]}"
                                    )
                
        else:
            print("No solution found.")
    
    class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
        """Print intermediate solutions."""

        def __init__(self, variables, limit=None):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.__variables = variables
            self.__solution_count = 0
            self.__solution_limit = limit

        def on_solution_callback(self):
            self.__solution_count += 1
            for v in self.__variables:
                
                print(f"[INFO] Solution {self.__solution_count}:", '%s=%i' % (v, self.Value(v)))
            if self.__solution_limit: # check if a limit is specified
                if self.__solution_count >= self.__solution_limit:
                    print('Stop search after %i solutions' % self.__solution_limit)
                    self.StopSearch()

        def solution_count(self):
            return self.__solution_count