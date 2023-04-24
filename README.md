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
r1_availability_slots = [
     {"slot_id" : 0, "start":0, "end":4},
     {"slot_id" : 1, "start":5, "end":10}
]
r2_availability_slots = [
     {"slot_id" : 0, "start":0, "end":10}
]
resource1 = Resource(id=1, availability_slots=r1_availability_slots)
resource2 = Resource(id=2, availability_slots=r2_availability_slots)
resource_group = ResourceGroup(id=1, resources=[resource1, resource2])
```
3. Define tasks with duration, and resource group:
```python
tasks = [Task(id=1, duration=3, resource_group=resource_group)]
```
4. Set the scheduling horizon:
```python
horizon = 50
```
5. Call the Scheduler function with tasks and horizon:
```python
s = Scheduler()
s.schedule(tasks)
```
If a solution is found, the program will print the optimal or feasible schedule with makespan, resource assignments, task start and end times, and durations.
```python
Makespan = 8.0
Assigned to resource 2
Task (0, 0): starts: 0, end: 8, duration: 8
```
