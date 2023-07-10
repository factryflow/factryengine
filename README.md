# PlanBee 🐝

If Plan A fails, dont worry there is always PlanBee! PlanBee is a Job Shop Scheduling algorithm module buzzing with features. Feed PlanBee with your tasks and resources and it quickly finds a solution. It employs the high-speed computation power of NumPy to achieve fast results. With PlanBee, Plan B becomes your Plan A!

## Features 🚀

- Define your own tasks and resources.
- Specify available windows for each resource.
- Indicate priority, duration, and necessary resources for each task.
- Solve your scheduling problems with a single function call!
- Get a detailed summary and visualization of the scheduling solution.

## Installation 🛠️

```sh
pip install planbee
```

## Usage 🐍

First, import the necessary modules:

```python
from PlanBee import Resource, Task, Scheduler
```

Then, define your resources:

```python
resource1 = Resource(id=1, available_windows=[(0, 10), (15, 20)])
resource2 = Resource(id=2, available_windows=[(5, 20)])
```

And your tasks:

```python
task1 = Task(id=1, duration=5, priority=1, resources=[resource1, resource2], resource_count=1)
task2 = Task(id=2, duration=3, priority=2, resources=[resource1], predecessors=[task1], resource_count=1)
```

Finally, use the Scheduler to solve:

```python
scheduler = Scheduler(tasks=[task1, task2])
result = scheduler.schedule()
```

## Visualization 📊

PlanBee provides a function to plot your schedule:

```python
result.plot_resource_plan()
```

## Contributions 💡

Contributions are always welcome! See `CONTRIBUTING.md` for ways to get started.

## License 📄

This project is licensed under the terms of the [MIT license](LICENSE.md).
