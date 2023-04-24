from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from ortools.sat.python import cp_model


@dataclass
class Resource:
    resource_id: int
    availability_slots: List = field(default_factory=list)

    def __eq__(self, other):
        if isinstance(other, Resource):
            return self.resource_id == other.resource_id
        return NotImplemented
    
    def __hash__(self):
        return hash(self.resource_id)

@dataclass
class ResourceGroup:
    resource_group_id: int
    resources: List[Resource]

    def __eq__(self, other):
        if isinstance(other, ResourceGroup):
            return self.resource_group_id == other.resource_group_id
        return NotImplemented
    
    def __hash__(self):
        return hash(self.resource_group_id)
    
@dataclass
class Task:
    task_id: int
    duration: int
    priority: int
    resource_group: ResourceGroup
    predecessors: Optional[List['Task']] = field(default_factory=list)

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.task_id == other.task_id
        return NotImplemented

    def __hash__(self):
        return hash(self.task_id)

def Scheduler(tasks, horizon):

    resource_groups = set(task.resource_group for task in tasks)
    resources = set(resource for resource_group in resource_groups for resource in resource_group.resources) 
    
    availability_slots = [
        {"slot_id" : 0, "start":0, "end":4}, # duration 1
        {"slot_id" : 1, "start":5, "end":10}, # duration 2
        {"slot_id" : 2, "start":15, "end":20}, # duration 3
        {"slot_id" : 3, "start":22, "end":50} # duration 4
    ]

    model = cp_model.CpModel()

    # subtaks 
    vars = {task.task_id: [] for task in tasks}

    # Create subtasks for each availability slot
    for task in tasks:
        task_id = task.task_id
        task_duration = task.duration
        subtask_durations = []
        for slot in availability_slots:

            # create variables
            start = model.NewIntVar(slot["start"], slot["end"], f'start_{slot["start"]}')
            end = model.NewIntVar(slot["start"], slot["end"], f'end_{slot["end"]}')
            duration = model.NewIntVar(0, task_duration, f'duration_{slot["slot_id"]}')
            subtask_durations.append(duration)
            interval = model.NewIntervalVar(start, duration, end, f'interval_{task_id, slot["slot_id"]}')

            # Create a BoolVar to check if slot is task start
            is_task_start = model.NewBoolVar(f'task_start_{task_id}_{slot["slot_id"]}')
            model.Add(sum(subtask_durations) == duration).OnlyEnforceIf((is_task_start))
            model.Add(sum(subtask_durations) != duration).OnlyEnforceIf(is_task_start.Not())

            # Create a BoolVar to check if task has started
            is_in_progress = model.NewBoolVar(f'task_started_{task_id}_{slot["slot_id"]}')
            model.Add(sum(subtask_durations) > 0).OnlyEnforceIf(is_in_progress)
            model.Add(sum(subtask_durations) == 0).OnlyEnforceIf(is_in_progress.Not())

            # Create a BoolVar to check if task has ended
            is_task_end = model.NewBoolVar(f'task_ended_{task_id}_{slot["slot_id"]}')
            model.Add(sum(subtask_durations) == task_duration).OnlyEnforceIf(is_task_end)
            model.Add(sum(subtask_durations) != task_duration).OnlyEnforceIf(is_task_end.Not())

            # Ensure subtasks are continuous 
            # if task is in progress it should fill the whole slot
            model.Add(start == slot["start"]).OnlyEnforceIf((is_in_progress,is_task_end.Not(), is_task_start.Not()))
            model.Add(end == slot["end"]).OnlyEnforceIf((is_in_progress,is_task_end.Not(), is_task_start.Not()))

            # if task is starting but not ending
            model.Add(end == slot["end"]).OnlyEnforceIf((is_task_start, is_task_end.Not()))

            # if task is ending but not starting
            model.Add(start == slot["start"]).OnlyEnforceIf((is_task_end,is_task_start.Not()))        

            # Add the subtask to the list
            vars[task_id].append({
                "subtask_id" : slot["slot_id"],
                "start" : start,
                "duration" : duration,
                "end" : end,
                "interval" : interval,
                "slot_id": slot["slot_id"],
                "is_task_start" : is_task_start,
                "is_in_progress" : is_in_progress,
                "is_task_end" : is_task_end
            })

        # Add constraint to enforce the sum of durations of subtasks to be equal to task_duration
        model.Add(sum(subtask_vars["duration"] for subtask_vars in vars[task_id]) == task_duration)

    # unpack vars
    all_intervals = []
    all_ends = []

    for task_id, subtasks in vars.items():
        for subtask in subtasks:
            all_intervals.append(subtask["interval"])
            all_ends.append(subtask["end"])


    # model.AddNoOverlap(all_intervals)

    # resource implementation
    x = {}
    for task in tasks:
        for resource in task.resource_group.resources:
            x[resource, task.task_id] = model.NewBoolVar(f'x[{resource},{task.task_id}]')

    for task in tasks:
        model.AddExactlyOne([x[resource, task.task_id] for resource in task.resource_group.resources])

    # Add NoOverlap constraint for each resource.
    resource_intervals = {resource: [] for resource in resources}
    for task in tasks:
        for subtask in vars[task.task_id]:
            for resource in task.resource_group.resources:
                optional_interval = model.NewOptionalIntervalVar(
                    start = subtask["start"], 
                    size = subtask["duration"], 
                    end = subtask["end"], 
                    is_present = x[resource, task.task_id],
                    name = f'resource_interval_{resource}_{task.task_id, subtask["subtask_id"]}'
                    )
                resource_intervals[resource].append(optional_interval)

    for resource in resources:
        model.AddNoOverlap(resource_intervals[resource])

    # object var to reduce makespan

    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, all_ends)
    model.Minimize(obj_var)

    # Create a solver and solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Check the solver status and print the solution
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Makespan = {solver.ObjectiveValue()}")
        for task_id, subtasks in vars.items():
            for resource in task.resource_group.resources:
                if solver.BooleanValue(x[resource, task_id]):
                    print(f"Assigned to resource {resource.resource_id}")
            for subtask in subtasks:
                if solver.Value(subtask['duration']) > 0:
                    print(
                        f"Task {(task_id, subtask['subtask_id'])}: "
                        f"starts: {solver.Value(subtask['start'])}, "
                        f"end: {solver.Value(subtask['end'])}, "
                        f"duration: {solver.Value(subtask['duration'])}, "
                        f"bools: {[solver.Value(subtask[field]) for field in ['is_task_start', 'is_in_progress', 'is_task_end']]}"
                        )
    else:
        print("No solution found.")