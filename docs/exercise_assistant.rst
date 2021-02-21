
Exercise Assistant
==================

A multi-agent environment that involves an athlete performing an exercise and a exercise assistant that provides feedback to the athlete. Both agents share the common goal of the athlete completing as many reps of an exercise without the athlete over-exerting themself.

An episode involves the athlete performing a fixed number of sets (MAX_SETS) of the exercise. After each set the athletes is complete their energy is replenished a random amount. The episode ends after all sets are complete or the athlete becomes over-exerted.

Each agent alternates performing actions:


1. The exercise-assistant, after observing athlete energy level, sends their signal to the athlete
2. The athlete, after observing coaches signal, chooses action (either perform rep or end set)


The key interesting aspect of this environment is the asymmetry of information and control. The Exercise Assistant has perfect knowledge of the current state, i.e. via access to advanced sensors, while the athlete has only noisy observations of the current state. Meanwhile, the athlete has full direct control over the state of the environment, while the exercise assistant can only control the state via the signals it sends to the athlete. This creates an interesting dynamic which depends a lot on how much the athlete trusts their percieved energy level versus what the exercise assistant is recommending.


State Space
~~~~~~~~~~~

- **Athlete Energy Level**: float in [0.0, 1.0]
    - The proportion of energy the athlete has remaining, if this reaches 0.0, the athlete is overexerted.
- **Sets Completed**: int in [0, MAX_SETS]
    - The number of sets completed.


Starting State
--------------

**Athlete Energy Level** = uniform from [0.75, 1.0]
**Sets Completed** = 0

I.e. the athlete starts with energy level sampled uniformly at random from [0.75, 1.0]


Action Space
~~~~~~~~~~~~




Observation Space and function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




Transition
~~~~~~~~~~




Reward
~~~~~~
