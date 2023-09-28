# factryengine

`factryengine` is a Python package designed to simplify task scheduling by considering task priorities, resources, and predecessors. It allows you to easily create tasks and resources, define dependencies between tasks, and schedule them efficiently. With `factryengine`, you can ensure that tasks are executed in the correct order, taking into account their dependencies and resource availability, while also adhering to their priorities.

## Installation

Installing `factryengine` is straightforward and can be done using `pip`. Run the following command in your terminal:

```bash
pip install factryengine
```

This command fetches the latest version of `factryengine` from the Python Package Index (PyPI) and installs it on your system.

After installation, you can import and use `Resource`, `Task`, and `Scheduler` classes from `factryengine` in your Python scripts to model and solve your scheduling problems.

## Quick Start

Below is a basic example to get you started with `factryengine`:

```python
from factryengine import Task, Resource, Scheduler

# Creating a Resource object
resource = Resource(id=1, available_windows=[(0,10)])

# Creating Task objects
task1 = Task(id=1, duration=3, priority=2, resources=[[resource]])
task2 = Task(id=2, duration=5, priority=1, resources=[[resource]], predecessors=[task1])

# Creating a Scheduler object and scheduling the tasks
scheduler = Scheduler(tasks=[task1, task2])
scheduler_result = scheduler.schedule()
```

In this example, `task1` is scheduled before `task2` despite its lower priority, as `task2` is dependent on `task1`.
