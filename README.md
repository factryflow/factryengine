# Python Task Scheduler
This Python program is a task scheduler that takes a set of tasks and assigns them to resources within their respective resource groups. The program uses Google OR-Tools library to build a constraint programming model and solve it, minimizing the makespan of the tasks.

## Features
Define tasks with duration, priority, and resource groups
Define resources with availability slots
Schedule tasks to resources within their respective resource groups, minimizing the makespan
Output the optimal or feasible schedule if found
## Requirements
To run this program, you need to have the following packages installed:

Python 3.6 or later
ortools (pip install ortools)
## Usage
1. Import the necessary classes and functions:
```python
from scheduler import Resource, ResourceGroup, Task, Scheduler
```
2. Define resources and resource groups:
```python
resource1 = Resource(resource_id=1)
resource2 = Resource(resource_id=2)
resource_group = ResourceGroup(resource_group_id=1, resources=[resource1, resource2])
```
3. Define tasks with duration, priority, and resource group:
```python
task1 = Task(task_id=1, duration=3, priority=1, resource_group=resource_group)
task2 = Task(task_id=2, duration=4, priority=2, resource_group=resource_group)
tasks = [task1, task2]
```
4. Set the scheduling horizon:
```python
horizon = 50
```
5. Call the Scheduler function with tasks and horizon:
```python
Scheduler(tasks, horizon)
```
If a solution is found, the program will print the optimal or feasible schedule with makespan, resource assignments, task start and end times, and durations.
