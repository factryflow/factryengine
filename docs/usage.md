# Usage

The `factryengine` package contains three main components: `Resource`, `ResourceGroup`, `Assignment`, `Task`,  and `Scheduler`. Below is the documentation for using each of these components.

## Resource and ResourceGroup

The `Resource` class is used to represent a resource in the scheduling problem. Below is an example of how to create a `Resource` object.

```python
from factryengine import Resource

# Creating a Resource object
resource = Resource(
    id=1,
    available_windows=[(0, 5), (10, 15)],
)
```

**Attributes:**

- `id` (`int`): Unique identifier for the resource.
- `available_windows` (`list[tuple[int, int]]`): List of available windows for the resource represented as tuples of start and end times.

---

We can put multiple resources into a group which can be used for dynamic assignments.

```python
from factryengine import ResourceGroup

# create resource objects
operator1 = Resource(
    id=2,
    available_windows=[(0, 5), (10, 15)],
)
operator2 = Resource(
    id=3,
    available_windows=[(0, 5), (10, 15)],
)

# add them to resource group

operators = ResourceGroup([operator1, operator2])
```

## Assignment

Assignments are used for allocating resources to tasks dynamically. You dont need to hardcode the resources you can instead select a pool of resource that could do the task. The algorithm will select the subset which finishes the task the fastest.

```python
from factryengine import Assignment

assignment = Assignment(resource_groups=(operators), resource_count=1)
```
- `resource_groups` (`list[ResouceGroup]`): The resource groups which can be picked from. If multiple resource groups are added then the fastest completing resourcegroup will be selected.
- `resource_count` (`int`): The maximum number of resources which may be selected.
- `use_all_resources` (`bool`): if you want to allow to use all resources in a group then simple set use_all_resources to True.

## Task

The `Task` class is used to represent a task in the scheduling problem. Below is an example of how to create a `Task` object, reusing the `resource` object created above.

```python
from factryengine import Task

# Creating a Task object
task = Task(
    id=1,
    duration=5,
    priority=1,
    constraints=[resource],
    assigments=[assignment]
    predecessor_ids=[1],
    predecessor_delay=0,
)
```

**Attributes:**

- `id` (`int | str`): Unique identifier for the task.
- `duration` (`int`): Duration of the task with a constraint of being greater than 0.
- `priority` (`int`): Priority of the task with a constraint of being greater than 0.
- `constraints` (`set[Resource]`): Are resource which are required througout the task.
- `assignments` (list[Assignment]): Are used to dynamically assign resources to tasks. If 2 resources are allocated, the task will finish twice as fast.

- `predecessor_ids` (`list[int]`): List of predecessor task ids.
- `predecessor_delay` (`int`): Buffer time after the completion of predecessor tasks before the current task can commence, must be greater than 0.

## Scheduler

The `Scheduler` class is used to schedule tasks based on their priorities, durations, and resources. Below is an example of how to use the `Scheduler` class, reusing the `task` and `resource` objects created above.

```python
from factryengine import Scheduler

# Creating a Scheduler object
scheduler = Scheduler(tasks=[task], resources = [resource])

# Scheduling the tasks
scheduler_result = scheduler.schedule()
```

**Methods:**

- `schedule()`: This method schedules the tasks and returns a `SchedulerResult` object.

## SchedulerResult

The `SchedulerResult` class contains the results of the scheduling.

**Methods:**

- `to_dict()`: Converts the scheduling result to a dictionary.
- `to_dataframe()`: Converts the scheduling result to a pandas DataFrame.
- `summary()`: Provides a summary of the scheduling result.
- `plot_resource_plan()`: Plots the resource plan.
- `get_resource_intervals_df()`: Returns a DataFrame of resource intervals.

## Task Priority

In `factryengine`, task priority helps decide the order tasks are scheduled. Lower priority numbers are scheduled first. But, if a task depends on another (called a predecessor), the predecessor will be scheduled first, no matter its priority.

Here's a simple example:

```python
from factryengine import Task, Resource

# Task with lower priority
task1 = Task(id=1, duration=3, priority=2, constraints=[[resource]])

# Task with higher priority, but depends on task1
task2 = Task(id=2, duration=5, priority=1, constraints=[[resource]], predecessors_ids=[1])
```

In this case, even though `task2` has higher priority, `task1` will be scheduled first because `task2` depends on it.
