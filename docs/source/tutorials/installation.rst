.. _installation:

Installation
==============


Dependencies
--------------

This framework is tested to work under Python 3.7 or later.

The required dependencies:

* Python >= 3.7
* Gym >= 0.18
* NumPy >= 1.19
* Scipy >= 1.6
* highway-env == 1.0

For rendering:

* Pygame >= 2.0

.. _user-install:

User install instructions
--------------------------

Clone the repo:

.. code-block:: bash

    git clone -b master https://github.com/RDLLab/benevolent-deception-gym.git


Navigate into the root package directory:

.. code-block:: bash

    cd benevolent-deception-gym


Install BDGym (run the whichever command suits you purpose):

.. code-block:: bash

    # install base level without doc, test, etc dependencies
    pip install -e .

    # or install base level + doc dependencies
    pip install -e.[docs]

    # or install base level + docs + test dependencies
    pip install -e.[all]
