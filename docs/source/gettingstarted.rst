---------------
Getting started
---------------

Installation
------------

Installing OpenPile is as easy as installing well-known python packages such as Pandas or Numpy. This is done 
via the below pip command.

.. code-block:: console

    pip install openpile


**Below, you can find more help to get it installed on your local machine:**

#. Go to the official Python website at https://www.python.org/downloads/ and 
   download python for your operating system (Windows, macOS, or Linux).
   **Note that only python version ranging from 3.8 to 3.10 are valid for openpile to run.**
#. Run the installer and follow the prompts to complete the installation process.
#. After installing Python, open a terminal or command prompt and type python to verify that 
   Python is installed correctly. You should see the Python version number.
#. To start using Python packages, you can use the built-in package manager called pip. 
   Type pip or pip3 (depending on your installation) in the terminal to see if it's installed correctly.
#. To install the awesome OpenPile package, use the command:
   
   .. code-block::
    
      pip install openpile
      
   This will download and install the package and all its dependencies.


**Below, an alternative to the above should you want to use the Anaconda distribution:**

#. Download and install the latest version of Anaconda from https://www.anaconda.com/products/individual.
#. Open the Anaconda Navigator application and select the "Environments" tab.
#. Click the "Create" button to create a new environment and enter a name for it (e.g. python38).
#. Select the desired Python version from the dropdown menu (e.g. Python 3.8) and click the "Create" button.
#. To switch to a specific environment, select it from the list and click the "Activate" button.
#. To install a package in the environment using pip, open a terminal window by clicking the 
   "Open Terminal" button in the "Environments" tab.
#. Use the following command in the Anaconda command prompt or via the navigator to install a package using pip in the environment:
   
   .. code-block:: console

      pip install openpile


First model in OpenPile
-----------------------

Please refer to the :ref:`usage` section where examples can help you get started.

Philosophy
----------

these calculations can be as simple/generic as it gets, e.g. calculating the weight of a beam, its volume, or perform more advanced calculations with 1D Finite Element Method.

.. show a plot of the beam/pile 

.. ref to objects in API.


System of units
---------------

The unit system used in this library is the `International System of Units (SI) <https://en.wikipedia.org/wiki/International_System_of_Units>`_. 
The primary units used in OpenPile are kN (kilonewton) and m (meter). 
The documentation (e.g. docstrings) should inform the user just well enough. If there is any ambiguity, 
please create an issue so that we can solve this.


Coordinates System
------------------

The coordinate system in OpenPile follows right hand system as shown below:

.. figure:: _static/coordinates_system.png
    :width: 80%

    Coordinates system followed in OpenPile.



