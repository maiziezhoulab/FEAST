Installation
============

Published package
-----------------

Use a Python 3.11 environment for FEAST-py 1.0.6. The distribution is named
``FEAST-py`` on PyPI and the import name is ``FEAST``.

.. code-block:: bash

    python -m pip install "FEAST-py==1.0.6"
    python -c "from importlib.metadata import version; import FEAST; print(version('FEAST-py'), FEAST.__file__)"

Editable Conda installation
---------------------------

.. code-block:: bash

    git clone https://github.com/maiziezhoulab/FEAST
    cd FEAST
    conda env create -f environment.yml
    conda activate feast-py311-conda
    python -m pip install --no-deps -r requirements.txt
    python -m pip install --no-deps -e .

For an existing checkout, start at ``conda env create`` from the repository
root. The supplied environment selects a CUDA-enabled PyTorch build. The
default NumPy transport backend runs on CPU.

Documentation
-------------

Install Pandoc for notebook rendering. From the repository root, use the
Python 3.11 environment with FEAST installed.

.. code-block:: bash

    python -m pip install -r docs/requirements.txt
    python -m sphinx -b html docs/source docs/_build/html

Open ``docs/_build/html/index.html`` in a browser.
