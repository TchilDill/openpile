from openpile.soilmodels import Dunkirk_sand, Custom_pisa_sand
from openpile.utils.hooks import PISA_depth_variation

import pytest
import numpy as np
import math as m

from pydantic import ValidationError

@pytest.fixture
def get_dunkirk_sand_params_model_1():
    return {'sig':50, 'X':5, 'Dr':75, 'G0':50e3, 'D':6, 'L':20}

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
def get_depth_variation_funcs_dunkirk_sand_model_1(
    get_dunkirk_sand_params_model_1
):

    D = get_dunkirk_sand_params_model_1['D']
    L = get_dunkirk_sand_params_model_1['L']
    dr = get_dunkirk_sand_params_model_1['Dr']

    return {
        **PISA_depth_variation.dunkirk_sand_py_pisa_norm_param(D=D, L=L, Dr=dr),
        **PISA_depth_variation.dunkirk_sand_mt_pisa_norm_param(L=L, Dr=dr),
        **PISA_depth_variation.dunkirk_sand_Hb_pisa_norm_param(D=D, L=L, Dr=dr),
        **PISA_depth_variation.dunkirk_sand_Mb_pisa_norm_param(D=D, L=L, Dr=dr),
    }

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

    assert p == pytest.approx(p_custom, rel=1e-3, abs=1e-3)


    #mt curves
    theta, m = create_dunkirk_sand_model_1.mt_spring_fct(
        **params
    )
    theta_custom, m_custom = create_custom_dunkirk_sand_model_1.mt_spring_fct(
        **params
    )

    assert m == pytest.approx(m_custom, rel=1e-3, abs=1e-3)

    #Hb curves
    y, Hb = create_dunkirk_sand_model_1.Hb_spring_fct(
        **params
    )
    y_custom, Hb_custom = create_custom_dunkirk_sand_model_1.Hb_spring_fct(
        **params
    )

    assert Hb == pytest.approx(Hb_custom, rel=1e-3, abs=1e-3)

    #Mb curves
    theta, Mb = create_dunkirk_sand_model_1.Mb_spring_fct(
        **params
    )
    theta_custom, Mb_custom = create_custom_dunkirk_sand_model_1.Mb_spring_fct(
        **params
    )

    assert Mb == pytest.approx(Mb_custom, rel=1e-3, abs=1e-3)
