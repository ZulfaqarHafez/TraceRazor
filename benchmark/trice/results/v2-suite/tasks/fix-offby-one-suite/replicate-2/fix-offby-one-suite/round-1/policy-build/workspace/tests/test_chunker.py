import pytest
from chunker import chunk

def test_even_split():
    assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

def test_remainder():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

def test_chunk_equals_len():
    assert chunk([1, 2, 3], 3) == [[1, 2, 3]]

def test_empty():
    assert chunk([], 3) == []

def test_no_overlap():
    flat = [x for c in chunk(list(range(10)), 3) for x in c]
    assert flat == list(range(10))

def test_bad_size():
    with pytest.raises(ValueError):
        chunk([1], 0)
