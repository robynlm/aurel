Advice
======

Working on an HPC
-----------------

While HPCs won't let you install packages globally, you can still use aurel. 
You just need to create a virtual environment and install aurel there.

* Create a virtual environment

.. code-block:: bash

   python -m venv ~/myenv
   source ~/myenv/bin/activate
   pip install aurel
   
and install any other packages you may use.

* If you want to use aurel in a **jupyter notebook**, add your virtual environment as a Jupyter kernel

.. code-block:: bash

   pip install ipykernel
   python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

then select `Python (myenv)` for your kernel and then in your notebook you can have

.. code-block:: python

   import aurel 

* If you want to use aurel in a **python script**, before running it, activate the environment

.. code-block:: bash

   source ~/myenv/bin/activate
   python myscript.py

then in your python script you can have

.. code-block:: python

   import aurel