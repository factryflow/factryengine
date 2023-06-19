import numpy as np
import pytest
from pyplanpro.scheduler.heuristic_solver.task_allocator import TaskAllocator


@pytest.fixture
def task_allocator():
    return TaskAllocator()


@pytest.mark.parametrize(
    "task_duration, resource_count, expected",
    [
        (5, 1, {1: (0, 2.5)}),
        (15, 1, {1: (0, 20)}),
        (30, 2, {0: (20, 30), 1: (0, 30)}),
        (40, 2, None),
    ],
)
def test_find_earliest_solution(
    task_allocator, task_duration, resource_count, expected
):
    matrix = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 10.0],
            [10.0, 10.0, 10.0],
            [20.0, 0.0, 15.0],
            [30.0, 10.0, 20.0],
        ]
    )
    resource_ids = [0, 1]
    result = task_allocator.find_earliest_solution(
        matrix=matrix,
        task_duration=task_duration,
        resource_ids=resource_ids,
        resource_count=resource_count,
    )
    assert result == expected


test_case_values = [
    # trim both intervals
    (
        "No Resets",
        [
            np.array([[0, 10, 10, 0], [20, 30, 10, 0]]),
            np.array([[5, 15, 10, 0], [20, 30, 10, 0]]),
        ],
        np.array(
            [
                [0, 0, 0],
                [5, 5, 0],
                [10, 10, 5],
                [15, 10, 10],
                [20, 10, 10],
                [30, 20, 20],
            ]
        ),
    ),
    (
        "With resets and high durations",
        [
            np.array([[0, 10, 20, 0], [20, 30, 10, -1]]),
            np.array([[5, 15, 10, 0], [20, 30, 10, -1]]),
            np.array([[0, 30, 60, 0]]),
        ],
        np.array(
            [
                [0, 0, 0, 0],
                [5, 10, 0, 10],
                [10, 20, 5, 20],
                [15, 20, 10, 30],
                [20, 0, 0, 40],
                [30, 10, 10, 60],
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "test_name, resource_windows_dict, expected",
    test_case_values,
    ids=[case[0] for case in test_case_values],
)
def test_create_matrix(test_name, task_allocator, resource_windows_dict, expected):
    result = task_allocator.create_matrix(resource_windows_dict)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.array([0, 1, 3, 0, 1, 1]), 3),
        (np.array([0, 2, 3, 4, 5]), 0),
        (np.array([2, 3, 4, 5, 6]), 0),
    ],
)
def test_get_window_start_index(task_allocator, array, expected):
    result = task_allocator._get_window_start_index(array)
    assert np.array_equal(result, expected)


def test_mask_smallest_except_k_largest(task_allocator):
    array = np.array(
        [
            [5, 0, 10],
            [10, 30, 20],
            [1, 1, 1],
        ]
    )
    result = task_allocator._mask_smallest_except_k_largest(array, 2).mask
    expected = np.array(
        [
            [False, True, False],
            [True, False, False],
            [True, False, False],
        ]
    )
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.array([0, 1, 5, -1, 10]), [0, 1, 6, 0, 10]),
        (np.array([-1, 2, 3, 0, 4]), [0, 2, 5, 5, 9]),
        (np.array([0, -1, 2, 4, -1]), [0, 0, 2, 6, 0]),
    ],
)
def test_cumsum_reset_at_minus_one(task_allocator, array, expected):
    result = task_allocator._cumsum_reset_at_minus_one(array)
    assert np.array_equal(result, expected)
