from PyPlanPro import Resource, Task, ResourceGroup
from PyPlanPro.scheduler.heuristic_solver import HeuristicSolver
from intervaltree import IntervalTree, Interval
import pytest
import copy

# Sample data for testing
resources = [
    Resource(id=0, availability_slots=[(0,10),(20,30)]),
    Resource(id=1, availability_slots=[(2,10), (12,20)])
]

rg = ResourceGroup(id=1, resources=resources)
tasks = [
    Task(id=0, duration=10, priority=1, resource_group=rg),
    Task(id=1, duration=5, priority=2, resource_group=rg, predecessor_delay=5),
    Task(id=2, duration=10, priority=2, resource_group=rg),
    Task(id=3, duration=5, priority=3, resource_group=rg),
    Task(id=4, duration=1, priority=4, resource_group=rg)
]
tasks[0].predecessors = [tasks[3]]
tasks[1].predecessors = [tasks[4]]
tasks[2].predecessors = [tasks[4]]

solver = HeuristicSolver(tasks,resources)

@pytest.mark.parametrize("availability_slots, task_duration, latest_start, expected", [
    ([(0, 10), (20, 30)], 10, 0, (0, 10, {Interval(0, 10)})),
    ([(0, 10), (20, 30)],10, 2, (2, 22, {Interval(2, 10), Interval(20, 22)})),
    ([(0, 10), (20, 30)],100, 0, (None, None, None)),

    # Add more test scenarios here
])
def test_get_task_earliest_start_end(availability_slots, task_duration, latest_start, expected):
    interval_tree = IntervalTree.from_tuples(availability_slots)
    solver = HeuristicSolver([], [])
    result = solver._get_task_earliest_start_end(
        interval_tree=interval_tree,
        task_duration=task_duration,
        latest_start_time=latest_start
    )
    assert result == expected

def test_get_task_order():
    ordered_task_ids = solver._get_task_order(tasks)
    assert ordered_task_ids == [3, 0, 4, 2, 1]

@pytest.mark.parametrize("task, expected_resource, max_start_time", [
    (tasks[0], resources[0], 0),
    (tasks[0], resources[1], 2),
])
def test_get_fastest_resource(task, expected_resource, max_start_time):
    rit = solver._get_resource_interval_trees(resources)
    fastest_resource = solver._get_fastest_resource(task, rit, max_start_time)
    assert fastest_resource.get("resource") == expected_resource

def test_update_resource_interval_trees():
    resource_id = 0
    interval_tree_index = 0
    task_start = 4
    task_end = 6
    rits = solver._get_resource_interval_trees(resources)
    rits_old = copy.deepcopy(rits)
    # update resource interval trees
    solver._update_resource_interval_trees(
    rits, resource_id, interval_tree_index, task_start, task_end)

    assert rits[resource_id] != rits_old[resource_id] ,"Updated does not equal original"
    assert len(rits[resource_id]) == (len(rits_old[resource_id]) + 1), "Update adds 1 more slot"
    # check no other resource intervals are modified
    del rits_old[resource_id]
    del rits[resource_id]
    assert rits == rits_old, "Are identical if change is deleted"