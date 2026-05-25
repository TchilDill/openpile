from openpile.core import misc
import pytest

def test_from_list2x_parse_top_bottom():
    # check if integer is passed
    t, b = misc.from_list2x_parse_top_bottom(2)
    assert t == 2
    assert b == 2
    # check if integer is passed
    t, b = misc.from_list2x_parse_top_bottom(5.0)
    assert t == 5.0
    assert b == 5.0
    # check if valid list is passed
    t, b = misc.from_list2x_parse_top_bottom([0.4, 50.6])
    assert t == 0.4
    assert b == 50.6

#create a matrix of test for misc.conic where the input varies and where we check that the output is a vector of ascending values
@pytest.mark.parametrize("Xbar", [0.1, 1, 5, 10, 50, 100, 1000])
@pytest.mark.parametrize("k", [0, 0.1, 0.5, 1, 10, 100])
@pytest.mark.parametrize("n", [0, 0.2, 0.8, 0.99])
@pytest.mark.parametrize("Ybar", [0.1, 1, 5, 10, 50, 100, 1000])
def test_conic(Xbar, k, n, Ybar):
    # calculate the conic values
    x, y = misc.conic(Xbar, n, k, Ybar, 100)
    # check that the output is a vector of ascending values
    assert all(x[i] <= x[i + 1] for i in range(len(x) - 1))
    assert all(y[i] <= y[i + 1] for i in range(len(y) - 1))

def test_var_to_str():
    print_out = misc.var_to_str(11)
    assert print_out == 11
    print_out = misc.var_to_str([11, 21.2])
    assert print_out == "11-21.2"
