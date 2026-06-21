import pytest
from stats import median

def test_odd():
    assert median([3, 1, 2]) == 2

def test_even():
    assert median([4, 1, 3, 2]) == 2.5

def test_single():
    assert median([7]) == 7

def test_unsorted_floats():
    assert median([2.5, 0.5, 1.5]) == 1.5

def test_empty_raises():
    with pytest.raises(ValueError):
        median([])

def test_input_not_mutated():
    xs = [3, 1, 2]
    median(xs)
    assert xs == [3, 1, 2]
