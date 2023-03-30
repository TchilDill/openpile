-----
Usage
-----

The package allows a quick input by the user (given in this section) and quick calculation. 

Jupyter Notebooks/IPython are recommended platforms to learn how to use openpile as it provides 
an interactive experience. 

Example 1 - Create a pile 
=========================

A pile can be created in the following way in openpile. It is strongly advised to use 
the Pile.create() constructor to ensure that data validation and post-processing of the object is performed.

.. code-block:: python

    from openpile.construct import Pile

    # Create a pile instance with two sections of respectively 10m and 30m length.
    pile = Pile.create(name = "",
            kind='Circular',
            material='Steel',
            top_elevation = 0,
            pile_sections={
                'length':[10,30],
                'diameter':[7.5,7.5],
                'wall thickness':[0.07, 0.08],
            }
        )

Once the pile (object) is created, the user can use its properties and methods to interact with it. 
A simple view of the pile can be extracted by printing the object as below 

.. code-block:: python
    
    # Print the pile data
    print(pile)
        Elevation [m]  Diameter [m]  Wall thickness [m]  Area [m2]     I [m4]
    0            0.0           7.5                0.07   1.633942  11.276204
    1          -10.0           7.5                0.07   1.633942  11.276204
    2          -10.0           7.5                0.08   1.864849  12.835479
    3          -40.0           7.5                0.08   1.864849  12.835479

The user can also extract easily the pile's length, its elevations and other properties.
Please see the :py:class:`openpile.construct.Pile`


As of now, only a circular pile can be modelled in openpile, however the user can bypass 
the construcutor by updating the pile's properties governing the pile's behaviour under 
axial or lateral loading.

.. code-block:: python

    # Override young's modulus
    pile.E = 250e6
    # Check young's modulus (value in kPa)
    print(pile.E)
    250000000.0
    # Override second moment of area across whole pile [in meters^4]
    pile.I = 1.11
    # Check updated second moment of area
    print(pile)
        Elevation [m]  Diameter [m] Wall thickness [m] Area [m2]  I [m4]
    0            0.0           7.5               <NA>  1.633942    1.11
    1          -10.0           7.5               <NA>  1.633942    1.11
    2          -10.0           7.5               <NA>  1.864849    1.11
    3          -40.0           7.5               <NA>  1.864849    1.11
    # Override pile's width or pile's diameter [in meters]
    pile.width = 2.22
    # Check updated width or diameter
    print(pile)
        Elevation [m]  Diameter [m] Wall thickness [m] Area [m2]  I [m4]
    0            0.0          2.22               <NA>  1.633942    1.11
    1          -10.0          2.22               <NA>  1.633942    1.11
    2          -10.0          2.22               <NA>  1.864849    1.11
    3          -40.0          2.22               <NA>  1.864849    1.11
    # Override pile's area  [in meters^2]
    pile.area = 1.0
    # Check updated width or diameter
    print(pile)
        Elevation [m]  Diameter [m] Wall thickness [m] Area [m2]  I [m4]
    0            0.0          2.22               <NA>       1.0    1.11
    1          -10.0          2.22               <NA>       1.0    1.11
    2          -10.0          2.22               <NA>       1.0    1.11
    3          -40.0          2.22               <NA>       1.0    1.11


Example 2 - Calculate and plot a p-y curve 
==========================================

openpile allows for quick access to soil curves. The below example shows
how one can quickly calculate a soil spring at a given elevation and plot it.

The different curves available can be found in:

* :py:mod:`openpile.utils.py_curves`
* :py:mod:`openpile.utils.mt_curves`
* :py:mod:`openpile.utils.tz_curves`

.. code-block:: python
    
    import matplotlib.pyplot as plt
    from openpile.utils.py_curves import api_sand

    p, y = api_sand(sig=50, # vertical stress in kPa 
                    X = 5, # depth in meter
                    phi = 35, # internal angle of friction 
                    D = 5, # the pile diameter
                    below_water_table=True, # use initial subgrade modulus under water
                    Neq=1, # static curve
                    )

    plt.plot(y,p)
    plt.ylabel('p [kN/m/m]')
    plt.xlabel('y [m]')

.. image:: _static/usage/pycurves/api_sand_example_build.png
    :width: 65%    



Example 3 - Create a soil profile's layer 
=========================================


Example 4 - Create a soil profile 
=================================
