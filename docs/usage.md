# Usage

The `factryengine` package contains three main components: `Resource`, `Task`, and `Scheduler`. Below is the documentation for using each of these components.

## Resource

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

## Task

The `Task` class is used to represent a task in the scheduling problem. Below is an example of how to create a `Task` object, reusing the `resource` object created above.

```python
from factryengine import Task

# Creating a Task object
task = Task(
    id=1,
    duration=5,
    priority=1,
    resources=[[resource]],
    resource_count=1,
    predecessors=[],
    predecessor_delay=0,
)
```

**Attributes:**

- `id` (`int | str`): Unique identifier for the task.
- `duration` (`int`): Duration of the task with a constraint of being greater than 0.
- `priority` (`int`): Priority of the task with a constraint of being greater than 0.
- `resources` (`list[set[Resource]]`): List of sets of resources required by the task.
- `resource_count` (`int | str`): Number of resources required by the task.
- `predecessors` (`list["Task"]`): List of predecessor tasks.
- `predecessor_delay` (`int`): Buffer time after the completion of predecessor tasks before the current task can commence, must be greater than 0.
  
!!! tip
    Resource count allows you to specifiy to use all resources specified:
    ```python
    resource_count = "all"
    ```

## Scheduler

The `Scheduler` class is used to schedule tasks based on their priorities, durations, and resources. Below is an example of how to use the `Scheduler` class, reusing the `task` object created above.

```python
from factryengine import Scheduler

# Creating a Scheduler object
scheduler = Scheduler(tasks=[task])

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
task1 = Task(id=1, duration=3, priority=2, resources=[[resource]])

# Task with higher priority, but depends on task1
task2 = Task(id=2, duration=5, priority=1, resources=[[resource]], predecessors=[task1])
```

In this case, even though `task2` has higher priority, `task1` will be scheduled first because `task2` depends on it.
