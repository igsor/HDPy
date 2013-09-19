"""
This module combines Reinforcement Learning and Reservoir Computing by means
of an Actor-Critic design. In Reinforcement Learning, the learning subject
is expressed through the agent while the teacher denoted as environment or plant.
At each time step, the agent chooses an action :math:`a_t`, which leads it from
state :math:`s_t` to state :math:`s_{t+1}`. The state information is provided
to the agent by the environment, together with a reward :math:`r_{t+1}` which
announces how good or bad the state is considered. Note that the reward cannot
be used as learning target, as it is not an error but merely a hint if the
agent goes into the right direction. Instead, the agent's goal is to collect
as much reward as possible with time. The Return expresses this by taking
future rewards into account:

.. math::
    R_t = \sum\limits_{k=0}^T \gamma^k r_{t+k+1}

As it may not be meaningful to consider the whole future, the influence
of rewards is decreased the farther they are off. This is controlled through
the discount rate :math:`\gamma`. Further, experiments are often episodic
(meaning that they terminate somewhen). This is accounted for by summing
until the episode length :math:`T` [RL]_.

An Actor-Critic design splits the agent into two parts: The Actor decides on
the action, for which it is in turn criticised by the Critic. Meaning, that
the Critic learns long-time behaviour, i.e. approximates the Return, while
the Actor uses the Critic's approximation to select the action which maximizes
the Return in a single step. This module incorporates Reservoir Computing
as the Critic's function approximator [ESN-ACD]_.

"""
from rc import *
from rl import *
from plants import *
from analysis import *
from analysis_epuck import *
from analysis_puppy import *
from puppy import *
from inout import *
from hdp import *

import policies as policy
from policies_puppy import FRA, LRA, LRP # Deprecated, don't use like this but through policy.puppy.{FRA,LRA,LRP}


#__all__ = ['']
