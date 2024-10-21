import numpy as np
import pytest
from factryengine.scheduler.heuristic_solver.task_allocator import Matrix, TaskAllocator
from factryengine.scheduler.heuristic_solver.window_manager import window_dtype


@pytest.fixture
def task_allocator():
    return TaskAllocator()


def test_can_allocate_task():
    pass


def test_solve_matrix():
    pass


def test_solve_task_end(task_allocator):
    resource_matrix = np.ma.array([[0, 0], [10, 10]])
    intervals = np.array([0, 10])
    task_duration = 10
    result_x, result_y = task_allocator._solve_task_end(
        resource_matrix, intervals, task_duration
    )
    assert result_x == 5
    assert np.array_equal(result_y, np.array([5, 5]))


def test_get_resource_intervals_continuous(task_allocator):
    # Test case continuous values 1 task 2 resources
    solution_resource_ids = np.array([1, 2])
    solution_intervals = np.array([0, 1])
    resource_matrix = np.ma.array([[0, 0], [1, 1]],  mask=[[False, False], [False, False]],)
    solution_matrix = Matrix(
        resource_ids=solution_resource_ids,
        intervals=solution_intervals,
        resource_matrix=resource_matrix,
    )
    result = task_allocator._get_resource_intervals(solution_matrix)
    expeceted = {1: [(0, 1)], 2: [(0, 1)]}
    assert result == expeceted

def test_get_resource_intervals_windowed(task_allocator):
    # Test case windowed values 1 task 1 resource
    solution_resource_ids = np.array([1])
    solution_intervals = np.array([0, 2, 3, 4])
    resource_matrix = np.ma.array([[0], [2], [2], [3]], mask=[[False], [False], [False], [False]],)
    solution_matrix = Matrix(
        resource_ids=solution_resource_ids,
        intervals=solution_intervals,
        resource_matrix=resource_matrix,
    )
    result = task_allocator._get_resource_intervals(solution_matrix)
    expeceted = {1: [(0, 2), (3, 4)]}
    assert result == expeceted


def test_create_matrix_from_resource_windows_dict(task_allocator):
    resource_windows_dict = {
        1: np.array([(0, 10, 10, 0), (20, 30, 10, -1)], dtype=window_dtype),
        2: np.array([(0, 10, 10, 0), (25, 30, 5, 0)], dtype=window_dtype),
    }
    result = task_allocator._create_matrix_from_resource_windows_dict(
        windows_dict=resource_windows_dict
    )
    expected_resource_ids = np.array([1, 2])
    expected_matrix = np.array(
        # interval, resource 1, resource 2
        [
            [0, 0, 0],
            [10, 10, 10],
            [20, -1, 0],
            [25, 5, 0],
            [30, 5, 5],
        ]
    )
    expected_intervals = expected_matrix[:, 0]
    expected_resource_matrix = np.ma.array(expected_matrix[:, 1:])

    assert np.array_equal(
        result.resource_ids, expected_resource_ids
    ), f"Expected resource_ids to be {expected_resource_ids}, but got {result.resource_ids}"

    assert np.array_equal(
        result.intervals, expected_intervals
    ), f"Expected intervals to be {expected_intervals}, but got {result.intervals}"

    assert np.array_equal(
        result.resource_matrix, expected_resource_matrix
    ), f"Expected resource_matrix to be {expected_resource_matrix}, but got {result.resource_matrix}"


@pytest.mark.parametrize(
    "array, k, expected",
    [
        (
            np.ma.array([[5, 0, 10], [10, 30, 20], [1, 1, 1]]),
            2,
            np.array(
                [[False, True, False], [True, False, False], [True, False, False]]
            ),
        ),
        (
            np.ma.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            1,
            np.array([[True, True, False], [True, True, False], [True, True, False]]),
        ),
        (
            np.ma.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]]),
            3,
            np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            ),
        ),
    ],
)
def test_mask_smallest_elements_except_top_k_per_row(
    task_allocator, array, k, expected
):
    result = task_allocator._mask_smallest_elements_except_top_k_per_row(array, k).mask
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
    print(f"Input: {array}, Result: {result}, Expected: {expected}")
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "y, x , expected",
    [
        (
            np.array([0, 1, 2, 3, 4]),
            np.array([0, 1, 2, 3, 4]),
            np.array([0, 1, 2, 3, 4]),
        ),
        (
            np.array([np.nan, np.nan, 0, 1, np.nan]),
            np.array([0, 1, 2, 3, 4]),
            np.array([0, 0, 0, 1, 0]),
        ),
        (
            np.array([0, np.nan, np.nan, np.nan, 4]),
            np.array([0, 1, 2, 3, 4]),
            np.array([0, 1, 2, 3, 4]),
        ),
    ],
)
def test_linear_interpolate_nan(y, x, expected):
    task_allocator = TaskAllocator()
    result = task_allocator._linear_interpolate_nan(y, x)
    print("result  :", result)
    print("expected:", expected)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "array, mask, expected",
    [
        (
            np.array([1, 3]),
            np.array([True, False, True]),
            np.array([1, np.nan, 3]),
        ),
        (
            np.array([]),
            np.array([False, False, False]),
            np.array([np.nan, np.nan, np.nan]),
        ),
        (
            np.array([1, 2, 3]),
            np.array([True, True, True]),
            np.array([1, 2, 3]),
        ),
    ],
)
def test_replace_masked_values_with_nan(array, mask, expected):
    task_allocator = TaskAllocator()
    result = task_allocator._replace_masked_values_with_nan(array, mask)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "array , expected",
    [
        (
            np.array([0, 1, 2, 3, 4]),
            np.array([0, 1, 1, 1, 1]),
        ),
        (
            np.array([0, 1, 2, 0, 1]),
            np.array([0, 1, 1, 0, 1]),
        ),
        (
            np.array([0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0]),
        ),
    ],
)
def test_diff_and_zero_negatives(array, expected):
    task_allocator = TaskAllocator()
    result = task_allocator._diff_and_zero_negatives(array)
    print("result  :", result)
    print("expected:", expected)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "array, expected",
    [
        # Full valid sequence without gaps
        (np.array([0, 1, 2, 3, 4]), (0, 4)),
        # Sequence with repeated values, expecting first valid segment
        (np.array([0, 3]), (0, 1)),  # This one might need revisiting if logic changes
    ],
)
def test_find_indexes(array, expected):
    task_allocator = TaskAllocator()
    result = task_allocator._find_indexes(array)
    print("result  :", result)
    print("expected:", expected)
    assert result == expected
