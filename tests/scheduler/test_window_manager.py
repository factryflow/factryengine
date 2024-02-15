import numpy as np
import pytest
from factryengine import Resource
from factryengine.scheduler.heuristic_solver.window_manager import (
    WindowManager,
    window_dtype,
)

# Test resources
resource1 = Resource(id=1, available_windows=[(1, 5), (7, 9)])
resource2 = Resource(id=2, available_windows=[(2, 6), (8, 10)])
resources = [resource1, resource2]


@pytest.fixture
def window_manager():
    resource1 = Resource(id=1, available_windows=[(1, 5), (7, 9)])
    resource2 = Resource(id=2, available_windows=[(2, 6), (8, 10)])
    resources = [resource1, resource2]
    return WindowManager(resources)


# Test case values
test_case_values = [
    ([resource1], {1: np.array([(1, 5, 4, 0), (7, 9, 2, 0)], dtype=window_dtype)}),
    ([resource2], {2: np.array([(2, 6, 4, 0), (8, 10, 2, 0)], dtype=window_dtype)}),
    (
        resources,
        {
            1: np.array([(1, 5, 4, 0), (7, 9, 2, 0)], dtype=window_dtype),
            2: np.array([(2, 6, 4, 0), (8, 10, 2, 0)], dtype=window_dtype),
        },
    ),
]


# Test the create_resource_windows_dict method
@pytest.mark.parametrize("resources, expected_output", test_case_values)
def test_create_resource_windows_dict(resources, expected_output):
    window_manager = WindowManager(resources)
    result = window_manager._create_resource_windows_dict()
    for key in expected_output:
        assert np.array_equal(result[key], expected_output[key])
    assert len(result) == len(expected_output)


# Test case values
test_case_values = [
    ([(1, 5), (7, 9)], np.array([(1, 5, 4, 0), (7, 9, 2, 0)], dtype=window_dtype)),
    ([(2, 6), (8, 10)], np.array([(2, 6, 4, 0), (8, 10, 2, 0)], dtype=window_dtype)),
]


# Test the windows_to_numpy method
@pytest.mark.parametrize("windows, expected_output", test_case_values)
def test_windows_to_numpy(windows, expected_output, window_manager):
    assert np.array_equal(window_manager._windows_to_numpy(windows), expected_output)


# # Test case values
test_case_values = [
    # trim end
    (
        "trim end",
        np.array([(1, 5, 4, 0), (8, 10, 2, 0)], dtype=window_dtype),
        (4, 8),
        np.array([(1, 4, 3, 0), (8, 10, 2, -1)], dtype=window_dtype),
    ),
    # trim start
    (
        "trim start",
        np.array([(1, 5, 4, 0), (8, 10, 2, 0)], dtype=window_dtype),
        (0, 4),
        np.array([(4, 5, 1, -1), (8, 10, 2, 0)], dtype=window_dtype),
    ),
    # trim both intervals
    (
        "trim both intervals",
        np.array([(1, 5, 4, 0), (8, 10, 2, 0)], dtype=window_dtype),
        (3, 9),
        np.array([(1, 3, 2, 0), (9, 10, 1, -1)], dtype=window_dtype),
    ),
    # trim no intervals
    (
        "trim no intervals - start",
        np.array([(1, 5, 4, 0), (8, 10, 2, 0)], dtype=window_dtype),
        (0, 1),
        np.array([(1, 5, 4, 0), (8, 10, 2, 0)], dtype=window_dtype),
    ),
    # trim no intervals
    (
        "trim no intervals - end",
        np.array([(1, 5, 4, 0), (8, 10, 2, 0)], dtype=window_dtype),
        (10, 15),
        np.array([(1, 5, 4, 0), (8, 10, 2, 0)], dtype=window_dtype),
    ),
    # trim no intervals
    (
        "trim no intervals - between",
        np.array([(1, 5, 4, 0), (8, 10, 2, 0)], dtype=window_dtype),
        (5, 8),
        np.array([(1, 5, 4, 0), (8, 10, 2, 0)], dtype=window_dtype),
    ),
    # trim all intervals
    (
        "trim all intervals",
        np.array([(1, 5, 4, 0), (8, 10, 2, 0)], dtype=window_dtype),
        (0, 20),
        np.array([], dtype=window_dtype),
    ),
    # split interval
    (
        "split interval",
        np.array([(1, 5, 4, 0), (8, 10, 2, 0)], dtype=window_dtype),
        (2, 4),
        np.array([(1, 2, 1, 0), (4, 5, 1, -1), (8, 10, 2, 0)], dtype=window_dtype),
    ),
    # trim single interval
    (
        "trim single interval",
        np.array([(1, 5, 4, 0), (8, 10, 2, 0), (11, 12, 1, 0)], dtype=window_dtype),
        (8, 10),
        np.array([(1, 5, 4, 0), (11, 12, 1, -1)], dtype=window_dtype),
    ),
]


# Test the trim_windows method
@pytest.mark.parametrize(
    "test_name, windows, trim_interval, expected",
    test_case_values,
    ids=[case[0] for case in test_case_values],
)
def test_trim_windows(
    test_name,
    windows,
    trim_interval,
    expected,
    window_manager,
):
    result = window_manager._trim_window(windows, trim_interval)
    print("result  :", result)
    print("expected:", expected)
    assert np.array_equal(result, expected)
