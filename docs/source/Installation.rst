Installation
============

MultiChat is developed based on Python libraries including Scanpy, PyTorch, and PyTorch Geometric. A clean conda environment is recommended to avoid dependency conflicts.

Set up a conda environment
--------------------------

.. code-block:: bash

   conda create -n env_MultiChat python=3.12
   conda activate env_MultiChat

Install PyTorch Geometric dependencies
--------------------------------------

PyTorch Geometric relies on system-specific libraries, so installing these dependencies before installing MultiChat is recommended.

.. code-block:: bash

   conda install pyg
   conda install conda-forge::pytorch_scatter
   conda install conda-forge::pytorch_cluster
   conda install conda-forge::pytorch_sparse

Install MultiChat
-----------------

The recommended installation method is PyPI.

.. code-block:: bash

   pip install scMultiChat

To install from source after the source code is released:

.. code-block:: bash

   git clone https://github.com/zcaiwei/MultiChat.git
   cd MultiChat-main
   pip install -r requirements.txt
   python setup.py build
   python setup.py install

Build this documentation locally
--------------------------------

The documentation environment is intentionally lighter than the full MultiChat runtime environment. It only needs Sphinx and notebook-rendering packages.

.. code-block:: bash

   conda create -n multichat-docs python=3.12
   conda activate multichat-docs
   pip install -r docs/requirements.txt
   sphinx-build -b html docs/source docs/build/html

After the build succeeds, open ``docs/build/html/index.html`` in a browser to preview the tutorial site.
