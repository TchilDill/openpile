from openpile.soilmodels import Dunkirk_sand, Custom_pisa_sand, Cowden_clay, Bothkennar_clay, Custom_pisa_clay
from openpile.utils.hooks import PISA_depth_variation

import pytest
import numpy as np
import math as m

from pydantic import ValidationError

@pytest.fixture
def get_dunkirk_sand_params_model_1():
    return {'sig':50, 'X':5, 'Dr':75, 'G0':50e3, 'D':6, 'L':20}

@pytest.fixture
def get_clay_params_model_1():
    return {'sig':50, 'X':5, 'Su':75, 'G0':50e3, 'D':6, 'L':20}


@pytest.fixture
def create_dunkirk_sand_model_1(
    get_dunkirk_sand_params_model_1
):
    # create a Dunkirk sand model with typical parameters (abritrary for testing)
    # we give this model a number: 1.
    return Dunkirk_sand(
        G0=get_dunkirk_sand_params_model_1['G0'],
        Dr=get_dunkirk_sand_params_model_1['Dr'],
    )

@pytest.fixture
def create_cowden_clay_model_1(
    get_clay_params_model_1
):
    # create a Dunkirk sand model with typical parameters (abritrary for testing)
    # we give this model a number: 1.
    return Cowden_clay(
        G0=get_clay_params_model_1['G0'],
        Su=get_clay_params_model_1['Su'],
    )

@pytest.fixture
def create_bothkennar_clay_model_1(
    get_clay_params_model_1
):
    # create a Dunkirk sand model with typical parameters (abritrary for testing)
    # we give this model a number: 1.
    return Bothkennar_clay(
        G0=get_clay_params_model_1['G0'],
        Su=get_clay_params_model_1['Su'],
    )


@pytest.fixture
def get_depth_variation_funcs_dunkirk_sand_model_1(
    get_dunkirk_sand_params_model_1
):

    D = get_dunkirk_sand_params_model_1['D']
    L = get_dunkirk_sand_params_model_1['L']
    dr = get_dunkirk_sand_params_model_1['Dr']

    return PISA_depth_variation.dunkirk_sand_pisa_norm_param(D=D, L=L, Dr=dr)

@pytest.fixture
def get_depth_variation_funcs_cowden_clay_model_1(
    get_clay_params_model_1
):

    D = get_clay_params_model_1['D']
    L = get_clay_params_model_1['L']

    return PISA_depth_variation.cowden_clay_pisa_norm_param(D=D, L=L)


@pytest.fixture
def create_custom_dunkirk_sand_model_1(
    get_dunkirk_sand_params_model_1,
    get_depth_variation_funcs_dunkirk_sand_model_1
):
    # we create the same model as above (dunkirk_sand_model_1) but using the Custom_pisa_sand class
    # this is useful to test that the Custom_pisa_sand class works as expected
    
    return Custom_pisa_sand(
        G0=get_dunkirk_sand_params_model_1['G0'],
        **get_depth_variation_funcs_dunkirk_sand_model_1
    )

@pytest.fixture
def create_custom_cowden_clay_model_1(
    get_clay_params_model_1,
    get_depth_variation_funcs_cowden_clay_model_1
):
    # we create the same model as above (dunkirk_sand_model_1) but using the Custom_pisa_sand class
    # this is useful to test that the Custom_pisa_sand class works as expected
    
    return Custom_pisa_clay(
        G0=get_clay_params_model_1['G0'],
        Su=get_clay_params_model_1['Su'],
        **get_depth_variation_funcs_cowden_clay_model_1
    )

def test_dunkirk_sand_model_1(
        create_dunkirk_sand_model_1, 
        create_custom_dunkirk_sand_model_1,
        get_dunkirk_sand_params_model_1):

    
    params = get_dunkirk_sand_params_model_1
    del params['Dr']
    del params['G0']
    params['layer_height'] = 2.0
    params['depth_from_top_of_layer'] = 1.2

    # py curves
    y, p = create_dunkirk_sand_model_1.py_spring_fct(
        **params
    )
    y_custom, p_custom = create_custom_dunkirk_sand_model_1.py_spring_fct(
        **params
    )

    assert y == pytest.approx(y_custom, rel=1e-3, abs=1e-3)
    assert p == pytest.approx(p_custom, rel=1e-3, abs=1e-3)


    #mt curves
    theta, m = create_dunkirk_sand_model_1.mt_spring_fct(
        **params
    )
    theta_custom, m_custom = create_custom_dunkirk_sand_model_1.mt_spring_fct(
        **params
    )

    assert theta == pytest.approx(theta_custom, rel=1e-3, abs=1e-3)
    assert m == pytest.approx(m_custom, rel=1e-3, abs=1e-3)

    #Hb curves
    y, Hb = create_dunkirk_sand_model_1.Hb_spring_fct(
        **params
    )
    y_custom, Hb_custom = create_custom_dunkirk_sand_model_1.Hb_spring_fct(
        **params
    )

    assert y == pytest.approx(y_custom, rel=1e-3, abs=1e-3)
    assert Hb == pytest.approx(Hb_custom, rel=1e-3, abs=1e-3)

    #Mb curves
    theta, Mb = create_dunkirk_sand_model_1.Mb_spring_fct(
        **params
    )
    theta_custom, Mb_custom = create_custom_dunkirk_sand_model_1.Mb_spring_fct(
        **params
    )

    assert theta == pytest.approx(theta_custom, rel=1e-3, abs=1e-3)
    assert Mb == pytest.approx(Mb_custom, rel=1e-3, abs=1e-3)


def test_cowden_clay_model_1(
        create_cowden_clay_model_1, 
        create_custom_cowden_clay_model_1,
        get_clay_params_model_1):

    
    params = get_clay_params_model_1
    del params['G0']
    del params['Su']
    params['layer_height'] = 2.0
    params['depth_from_top_of_layer'] = 1.2

    # py curves
    y, p = create_cowden_clay_model_1.py_spring_fct(
        **params
    )
    y_custom, p_custom = create_custom_cowden_clay_model_1.py_spring_fct(
        **params
    )

    assert y == pytest.approx(y_custom, rel=1e-3, abs=1e-3)
    assert p == pytest.approx(p_custom, rel=1e-3, abs=1e-3)


    #mt curves
    theta, m = create_cowden_clay_model_1.mt_spring_fct(
        **params
    )
    theta_custom, m_custom = create_custom_cowden_clay_model_1.mt_spring_fct(
        **params
    )

    assert theta == pytest.approx(theta_custom, rel=1e-3, abs=1e-3)
    assert m == pytest.approx(m_custom, rel=1e-3, abs=1e-3)

    #Hb curves
    y, Hb = create_cowden_clay_model_1.Hb_spring_fct(
        **params
    )
    y_custom, Hb_custom = create_custom_cowden_clay_model_1.Hb_spring_fct(
        **params
    )

    assert y == pytest.approx(y_custom, rel=1e-3, abs=1e-3)
    assert Hb == pytest.approx(Hb_custom, rel=1e-3, abs=1e-3)

    #Mb curves
    theta, Mb = create_cowden_clay_model_1.Mb_spring_fct(
        **params
    )
    theta_custom, Mb_custom = create_custom_cowden_clay_model_1.Mb_spring_fct(
        **params
    )

    assert theta == pytest.approx(theta_custom, rel=1e-3, abs=1e-3)
    assert Mb == pytest.approx(Mb_custom, rel=1e-3, abs=1e-3)
