
from openpile import construct
from openpile.soilmodels import API_clay
class TestPile:
    def test_main_constructor(self):
        # create a steel and circular pile
        pile = construct.Pile.create(kind='Circular',
                            name = "",
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
    def test_constructor(self):
        layer = construct.Layer(name='Soft Clay',
                        top=0,
                        bottom=-10,
                        weight=19,
                        lateral_model=API_clay(Su=[30,35], eps50=[0.01, 0.02], Neq=100), 
                    )
        assert 1==1

class TestMesh:
    def test_(self):
        pass