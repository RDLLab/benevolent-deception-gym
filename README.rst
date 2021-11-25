Benevolent Deception Gym
========================

|docs|

Multiagent Open AI gym environments that incoporate benevolent deception.

`Reference Docs <https://benevolent-deception-gym.readthedocs.io/en/latest/index.html>`_

Environments
~~~~~~~~~~~~

1. `Exercise-Assistant`
2. `Driver-Assistant`


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


Demo
~~~~

You can run a keyboard agent running the following command from the repo base directory::

  $ python demo/keyboard_agent.py ExerciseAssistantOA-v0


Substitute `ExerciseAssistantOA-v0` for any available benevolent-deception-gym environment. The docs contain a list of `Exercise Assistant <https://benevolent-deception-gym.readthedocs.io/en/latest/environments/exercise_assistant.html#environment-versions>`_ and `Driver Assistant <https://benevolent-deception-gym.readthedocs.io/en/latest/environments/driver_assistant.html#environment-versions>`_ versionsthat are available.


Acknowledgements
~~~~~~~~~~~~~~~~

This initiative was funded by the Department of Defence and the Office of National Intelligence under the AI for Decision Making Program, delivered in partnership with the NSW Defence Innovation Network.


Authors
~~~~~~~

**Jonathon Schwartz** - Jonathon.schwartz@anu.edu.au


License
~~~~~~~

`MIT`_ © 2020, Jonathon Schwartz

.. _MIT: LICENSE

.. |docs| image:: https://readthedocs.org/projects/benevolent-deception-gym/latest/badge/?version=latest
    :target: https://benevolent-deception-gym.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    :scale: 100%
