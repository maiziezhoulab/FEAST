Installation
============

Requirements
------------

- Python 3.11
- pip or conda

Conda environment (recommended)
-------------------------------

.. code-block:: bash

    git clone https://github.com/maiziezhoulab/FEAST
    cd FEAST
    conda env create -f environment.yml
    conda activate feast-py311-conda
    pip install --no-deps -r requirements.txt
    pip install --no-deps -e .

For an existing checkout, start at ``conda env create`` from the repository
root. The supplied environment selects a CUDA-enabled PyTorch build; the
compact tests and the default NumPy transport backend do not require a GPU.

Tests and documentation
-----------------------

Run the compact test suite from the repository root in the FEAST environment:

.. code-block:: bash

    PYTHONPATH=src python -m pytest tests

Build the documentation locally:

.. code-block:: bash

    python -m pip install -r docs/requirements.txt
    python -m sphinx -b html docs/source docs/_build/html

Open ``docs/_build/html/index.html`` in a browser. Generated documentation
under ``docs/_build/`` is ignored by Git; the source under ``docs/source/``
is tracked. Local validation reports under ``validation/`` are also ignored.
