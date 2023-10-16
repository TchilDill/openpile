---------------
Getting started
---------------

System of units
^^^^^^^^^^^^^^^

The unit system used in this library is the `International System of Units (SI) <https://en.wikipedia.org/wiki/International_System_of_Units>`_. 
The primary units used in OpenPile are kN (kilonewton) and m (meter). 
The documentation should show the units to the user. If there are any ambiguity, 
please create an issue so that we can solve this.


Installation
^^^^^^^^^^^^

Installing OpenPile is as easy as installing well-known python packages such as Pandas or Numpy. This is done 
via the below pip command.

.. code-block:: console

    pip install openpile


Some more help on installing python before OpenPile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Via the generic Python distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Go to the official Python website at https://www.python.org/downloads/ and 
   download python for your operating system (Windows, macOS, or Linux).
   **Note that only python version ranging from 3.7 to 3.10 are valid for openpile to run.**
#. Run the installer and follow the prompts to complete the installation process.
#. After installing Python, open a terminal or command prompt and type python to verify that 
   Python is installed correctly. You should see the Python version number.
#. To start using Python packages, you can use the built-in package manager called pip. 
   Type pip or pip3 (depending on your installation) in the terminal to see if it's installed correctly.
#. To install the awesome OpenPile package, use the command:
   
   .. code-block::
    
      pip install openpile
      
   This will download and install the package and all its dependencies.


Via the Anaconda distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Download and install the latest version of Anaconda from https://www.anaconda.com/products/individual.
#. Open the Anaconda Navigator application and select the "Environments" tab.
#. Click the "Create" button to create a new environment and enter a name for it (e.g. python37).
#. Select the desired Python version from the dropdown menu (e.g. Python 3.7) and click the "Create" button.
#. Repeat the previous steps to create environments for the other Python versions you want to install (e.g. 
   python38, python39, python310).
#. To switch to a specific environment, select it from the list and click the "Activate" button.
#. To install a package in the environment using pip, open a terminal window by clicking the 
   "Open Terminal" button in the "Environments" tab.
#. Use the following command in the Anaconda command prompt or via the navigator to install a package using pip in the environment:
   
   .. code-block:: console

      pip install openpile


First model in OpenPile
^^^^^^^^^^^^^^^^^^^^^^^

Please refer to the :ref:`usage` section where examples can help you get started.