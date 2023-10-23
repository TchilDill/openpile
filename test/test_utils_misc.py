from openpile.core import misc


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


def test_var_to_str():
    print_out = misc.var_to_str(11)
    assert print_out == 11
    print_out = misc.var_to_str([11, 21.2])
    assert print_out == "11-21.2"
