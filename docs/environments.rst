Benevolent Deception Environments
=================================


Percieved Effort
----------------

This environment involves two agents: an athlete and a coach. The athlete and coach share a common goal of trying to maximize the performance of the athlete on the given task, which in this case is running as far as possible without the athlete overexerting themselves.


State Space
~~~~~~~~~~~

The state space for this problem is simply the proportion of energy the athlete has remaining. This is initially 1.0 and decreases as the athlete moves down to a minimum of 0.0 at which point the athlete is considered to be overexerted.


Action Space
~~~~~~~~~~~~

**Athlete**: At each time step the athlete can choose to `move` or `stop`. The `move` action moves the athlete further at the cost of some energy, while the `stop` action ends the episode. The energy cost per step is stochastic.

**Coach**: At each time step the coach can signal either `green` or `red`, or display `no signal`. Unlike the athletes actions these signals have no direct effect on the state of the system but they can be used to communicate with the athlete.


Observation Space and function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both the athlete and the coach recieve an observation of the athletes energy level, however these observations are noisy. The accuracy of each agents observation can be chosen but in general if we mimic a hypothetical real world athlete-coach scenario we might expect that the athlete would overestimate how much energy they have exerted since it requires mental effort on their part, while the coach would more accurately or under estimate exertion.

The athlete additionally observes the signal of the coach.


Transition
~~~~~~~~~~

Each time the athlete uses the `move` action they lose some random amount of energy. The distribution of energy loss can be chosen, but an example would be a normal distribution with mean=1.0 and stdev=1.0. If the athlete uses the `stop` action the episode terminates. Finally, if the athletes energy reaches 0 or less then the episode terminates.


Reward
~~~~~~

For each `move` action performed the athlete and coach recieve a reward of 1. If the athlete performs the `stop` action the athlete and coach recieve a reward of 0. Lastly, if the energy level of the athlete goes to 0 or below then the athlete and coach recieve a large negative reward which can be chosen, e.g. -100 or -1000.



Nearly There
------------

Similar to `Percieved Effort' this environment involves two agents: an athlete and a coach. The athlete and coach share a common goal of the athlete reaching the finish line without the athlete overexerting themself.


State Space
~~~~~~~~~~~

The state space for this problem includes the state of the athlete and also the distance to the finish line.

The **athlete state** is simply the proportion of energy the athlete has remaining. This is initially 1.0 and decreases down to a minimum of 0.0 as the athlete moves. The athlete is considered to be overexerted if the athlete energy reaches 0.0.

The distance to the finish line is a float between 1.0 and 0.0, where 1.0 signifies 100% of distance remaining and 0.0 signifies that the athlete has reached the finish line.


Action Space
~~~~~~~~~~~~

**Athlete**: At each time step the athlete can choose to `move` or `stop`. The `move` action moves the athlete further towards the finish line at the cost of some energy, while the `stop` action ends the episode. The energy cost per step is stochastic, specifically it is Normally distributed with parameters **TODO**.

**Coach**: At each time step the coach can signal to the athlete their distance from the goal.


Observation Space and function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both the athlete and the coach recieve an observation of the athletes energy level, however these observations are noisy. The accuracy of each agents observation can be chosen but in general if we mimic a hypothetical real world athlete-coach scenario we might expect that the athlete would overestimate how much energy they have exerted since it requires mental effort on their part, while the coach would more accurately or under estimate exertion.

The coach also recieves an observation of the distance of the agent to the goal.

The athlete additionally observes the signal of the coach, which is a float between 1.0 and 0.0.


Transition
~~~~~~~~~~

Each time the athlete uses the `move` action they lose some random amount of energy and move some distance towards the goal. The distribution of energy loss can be chosen, but an example would be a normal distribution with mean=0.05 and stdev=0.025. If the athlete uses the `stop` action the episode terminates. Finally, if the athletes energy reaches 0.0 or less then the episode terminates.


Reward
~~~~~~

All actions of both the athlete and coach have a cost of 0. If the energy level of the athlete goes to 0 or below then the athlete and coach recieve a large negative reward which can be chosen, e.g. -100 or -1000. If the athlete reaches the finish line then they recieve a large reward, e.g. 100. The specific reward values can be chosen to experiment with different behavioural dynamics.
