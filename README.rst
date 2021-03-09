benevolent-deception-gym
========================

Multiagent environments that incoporate benevolent deception and implement Open AI Gym interface.


Credits
~~~~~~~

This environment is built on top of the awesome `Highway-Env <https://github.com/eleurent/highway-env>`_ created by Edouard Leurent.


Installation
~~~~~~~~~~~~

To install the benevolent-deception-gym environment, first you will need to clone the repo::


  $ git clone https://github.com/RDLLab/benevolent-deception-gym.git


Then next you will need to install via pip::

  $ cd benevolent-deception-gym
  $ pip install -e .


Documentation
~~~~~~~~~~~~~

Before it can be viewed the documentation must first be built. Firstly, if not already done, make sure documentation dependencies are installed::

  $ cd beneveolent-deception-gym
  $ pip install -e .[docs]


This should install the documentation dependencies, namely `Sphinx <https://www.sphinx-doc.org>`_. Once dependencies are installed the documentation can be built by running the following::

  $ cd docs
  $ make html


This should build all the documentation using `Sphinx`. Once the build is complete the docs can be accessed by opening ``benevolent-deception-gym/docs/build/html/index.html`` in your browser.


Authors
-------

**Jonathon Schwartz** - Jonathon.schwartz@anu.edu.au


License
-------

`MIT`_ © 2020, Jonathon Schwartz

.. _MIT: LICENSE
