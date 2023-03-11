
from openpile import construct

def test_from_list2x_parse_top_bottom():
    # check if integer is passed
    t, b = construct.from_list2x_parse_top_bottom(2)
    assert t == 2
    assert b == 2
    # check if integer is passed
    t, b = construct.from_list2x_parse_top_bottom(5.0)
    assert t == 5.0
    assert b == 5.0
    # check if valid list is passed
    t, b = construct.from_list2x_parse_top_bottom([0.4, 50.6])
    assert t == 0.4
    assert b == 50.6
    
def test_var_to_str():
    print_out = construct.var_to_str(11)
    assert print_out == 11
    print_out = construct.var_to_str([11,21.2])
    assert print_out == '11-21.2'
    

class TestPile:
    def test_main_constructor(self):
        # create a steel and circular pile
        pile = construct.Pile.create(kind='Circular',
                            material='Steel',
                            top_elevation = 0,
                            pile_sections={
                                'length':[10,30],
                                'diameter':[7.5,8.5],
                                'wall thickness':[0.07, 0.08],
                            }
                            )
        # check Young modulus is indeed Steel
        assert pile.E == 210e6
        # check even numbers of row for dataframe
        assert pile.data.values.shape[0] % 2 == 0

class TestLayer:
    def test_(self):
        layer = construct.Layer(name='Soft Clay',
                        top=0,
                        bottom=-10,
                        weight=9,
                        pymodel=construct.APIclay(Su=[30,35], eps50=[0.01, 0.02], Neq=100), 
                    )

class TestMesh:
    def test_(self):
        pass