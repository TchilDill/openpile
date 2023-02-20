import math as m
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from typing_extensions import Literal
from pydantic import BaseModel, Field, root_validator
from pydantic.dataclasses import dataclass
import matplotlib.pyplot as plt

import openpile.utils.graphics as graphics
import openpile.utils.validation as validation
from openpile.core.pile import Pile
from openpile.core.soilprofile import SoilProfile

class PydanticConfig:
    arbitrary_types_allowed = True


@dataclass(config=PydanticConfig)
class Mesh:
    """
    A class to create the mesh.

    The mesh is constructed based on the pile geometry and data primarily.
    As such it is possible to not feed any soil profile to the mesh, relying then on restrained degree(s) of freedom.
    
    The soil profile can be supplemented such that soil springs can be computed. After creating the mesh object, the soil springs can be
    generated via the `ssis()` method.
    
    Example
    -------
    
    """
    #: pile instance that the mesh should consider
    pile: Pile
    #: soil profile instance that the mesh should consider
    soil: Optional[SoilProfile] = None
    #: "EB" for Euler-Bernoulli or "T" for Timoshenko
    element_type: Literal['Timoshenko', 'EulerBernoulli'] = 'Timoshenko'
    #: x coordinates values to mesh as nodes
    x2mesh: List[float] = Field(default_factory=list)
    #: mesh coarseness, represent the maximum accepted length of elements
    coarseness: float = 0.5
    
    def get_structural_properties(self):
        try: 
            return self.element_properties
        except Exception as exc:
            print('Data not found. Please create mesh first.')
            raise Exception from exc

    def get_soil_properties(self):
        try:
            return self.soil_properties
        except Exception as exc:
            print('Data not found. Please create mesh first.')
            raise Exception from exc

    def _postinit(self):
        
        def get_coordinates() -> pd.DataFrame:
            
            # Primary discretisation over x-axis
            x = np.array([],dtype=np.float16)
            # add get pile relevant sections
            x = np.append(x,self.pile.data['Elevation [m]'].values)
            # add soil relevant layers and others
            if self.soil is not None:
                x = np.append(x,self.soil.data['Elevation [m]'].values)
            # add user-defined elevation
            x = np.append(x, self.x2mesh)
            
            # get unique values and sort in reverse order 
            x = np.unique(x)[::-1]
            
            # Secondary discretisation over x-axis depending on coarseness factor
            x_secondary = np.array([],dtype=np.float16)
            for i in range(len(x)-1):
                spacing = x[i] - x[i+1]
                new_spacing = spacing
                divider = 1
                while new_spacing > self.coarseness:
                    divider +=1
                    new_spacing = spacing/divider
                new_x = x[i] - (np.arange(start=1, stop=divider) * np.tile(new_spacing, (divider-1) ))                
                x_secondary = np.append(x_secondary, new_x)
                
            # assemble x- coordinates 
            x = np.append(x,x_secondary)
            x = np.unique(x)[::-1]
            
            # dummy y- coordinates
            y = np.zeros(shape= x.shape)
            
            #create dataframe coordinates
            nodes = pd.DataFrame(
                data = {
                    'x [m]': x,
                    'y [m]': y,
                }, dtype= float
            ).round(3)
            nodes.index.name = 'Node no.'
            
            element = pd.DataFrame(
                data = {
                    'x_top [m]': x[:-1],
                    'x_bottom [m]': x[1:],
                    'y_top [m]': y[:-1],
                    'y_bottom [m]': y[1:],
                }, dtype=float
            ).round(3)
            element.index.name = 'Element no.'
            
            return nodes, element
        
        # creates mesh coordinates
        self.nodes_coordinates, self.element_coordinates = get_coordinates()
        self.element_number = int(self.element_coordinates.shape[0])
        
        # creates element structural properties
        #merge Pile.data and self.coordinates
        self.element_properties = pd.merge_asof(left= self.element_coordinates.sort_values(by=['x_top [m]']),
                                                right= self.pile.data.drop_duplicates(subset='Elevation [m]',keep='last').sort_values(by=['Elevation [m]']),
                                                left_on='x_top [m]',
                                                right_on='Elevation [m]',
                                                direction='forward').sort_values(by=['x_top [m]'],ascending=False)
        #add young modulus to data
        self.element_properties['E [kPa]'] = self.pile.E
        #delete Elevation [m] column
        self.element_properties.drop('Elevation [m]', inplace=True, axis=1)
        #reset index
        self.element_properties.reset_index(inplace=True, drop=True)
        
        # create element soil properties
        if self.soil is None:
            self.soil_properties = None
        else:
            self.soil_properties = pd.DataFrame()

        # Initialise nodal global forces with link to nodes_coordinates (used for force-driven calcs)
        self.global_forces = self.nodes_coordinates.copy()
        self.global_forces['Px [kN]'] = 0
        self.global_forces['Py [kN]'] = 0
        self.global_forces['Mz [kNm]'] = 0

        # Initialise nodal global displacement with link to nodes_coordinates (used for displacement-driven calcs)
        self.global_disp = self.nodes_coordinates.copy()
        self.global_disp['Tx [m]'] = 0
        self.global_disp['Ty [m]'] = 0
        self.global_disp['Rz [rad]'] = 0
        
        # Initialise nodal global support with link to nodes_coordinates (used for defining boundary conditions)
        self.global_restrained = self.nodes_coordinates.copy()
        self.global_restrained['Tx'] = False
        self.global_restrained['Ty'] = False
        self.global_restrained['Rz'] = False

    def ssis(self) -> pd.DataFrame: 
        
        # function doing the work
        def create_springs():
            pass
        
        # main part of function
        try: 
            if self.soil is None:
                RuntimeError('No soil found. Please create the mesh first.')
            else: 
                self.soil_springs = create_springs()
        except Exception:
            print('No soil found. Please first create the mesh first.')
            raise
    
    def get_pointload(self, output = False, verbose = True):
        """_summary_

        _extended_summary_
        """
        out = ""
        try:
            for (idx, elevation, _, Px, Py, Mz) in self.global_forces.itertuples(name=None):
                if any([Px, Py, Mz]):
                    string = f"\nLoad applied at elevation {elevation} m (node no. {idx}): Px = {Px} kN, Py = {Py} kN, Mx = {Mz} kNm."
                    if verbose is True:
                        print(string)
                    out += f"\nLoad applied at elevation {elevation} m (node no. {idx}): Px = {Px} kN, Py = {Py} kN, Mx = {Mz} kNm."
            if output is True:
                return out
        except Exception:
            print('No data found. Please create the mesh first.')
            raise
        
    def set_pointload(self, elevation:float=0.0, Py:float=0.0, Px:float=0.0, Mz:float=0.0):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        elevation : float, optional
            _description_, by default 0.0
        Py : float, optional
            _description_, by default 0.0
        Px : float, optional
            _description_, by default 0.0
        Mz : float, optional
            _description_, by default 0.0
        """
    
        #identify if one node is at given elevation or if load needs to be split
        nodes_elevations = self.nodes_coordinates['x [m]'].values
        # check if corresponding node exist
        check = np.isclose(nodes_elevations ,np.tile(elevation, nodes_elevations.shape),atol=0.001)
        
        try:
            if any(check):
                #one node correspond, extract node
                node_idx = int(np.where(check == True)[0])
                # apply loads at this node
                self.global_forces.loc[node_idx,'Px [kN]'] = Px
                self.global_forces.loc[node_idx,'Py [kN]'] = Py
                self.global_forces.loc[node_idx,'Mz [kNm]'] = Mz
            else:
                if elevation > self.nodes_coordinates['x [m]'].iloc[0] or elevation < self.nodes_coordinates['x [m]'].iloc[-1]:
                    print("Load not applied! The chosen elevation is outside the mesh. The load must be applied on the structure.")
                else:
                    print("Load not applied! The chosen elevation is not meshed as a node. Please include elevation in `x2mesh` variable when creating the mesh.")
        except Exception:
            print("\n!User Input Error! Please create mesh first with the Mesh.create().\n")
            raise

    def set_support(self, elevation:float=0.0, Ty:bool=False, Tx:bool=False, Rz:bool=False):
            """_summary_

            _extended_summary_

            Parameters
            ----------
            elevation : float, optional
                _description_, by default 0.0
            Ty : float, optional
                _description_, by default 0.0
            Tx : float, optional
                _description_, by default 0.0
            Rz : float, optional
                _description_, by default 0.0
            """

            try:
                #identify if one node is at given elevation or if load needs to be split
                nodes_elevations = self.nodes_coordinates['x [m]'].values
                # check if corresponding node exist
                check = np.isclose(nodes_elevations ,np.tile(elevation, nodes_elevations.shape),atol=0.001)
                
                if any(check):
                    #one node correspond, extract node
                    node_idx = int(np.where(check == True)[0])
                    # apply loads at this node
                    self.global_restrained.loc[node_idx,'Tx'] = Tx
                    self.global_restrained.loc[node_idx,'Ty'] = Ty
                    self.global_restrained.loc[node_idx,'Rz'] = Rz
                else:
                    if elevation > self.nodes_coordinates['x [m]'].iloc[0] or elevation < self.nodes_coordinates['x [m]'].iloc[-1]:
                        print("Support not applied! The chosen elevation is outside the mesh. The support must be applied on the structure.")
                    else:
                        print("Support not applied! The chosen elevation is not meshed as a node. Please include elevation in `x2mesh` variable when creating the mesh.")
            except Exception:
                print("\n!User Input Error! Please create mesh first with the Mesh.create().\n")
                raise
        
    def plot(self, assign = False):
        fig = graphics.connectivity_plot(self)
        return fig if assign else None

    @classmethod
    def create(cls, pile: Pile, soil: Optional[SoilProfile] = None, element_type: Literal['Timoshenko', 'EulerBernoulli'] = 'Timoshenko', x2mesh: List[float] = Field(default_factory=list), coarseness: float = 0.5):
        
        obj = cls(pile=pile, soil=soil, element_type=element_type, x2mesh=x2mesh, coarseness=coarseness)
        obj._postinit()
        
        return obj
    
    def __str__(self):
        return self.element_properties.to_string()

