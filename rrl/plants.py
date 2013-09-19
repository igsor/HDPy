"""
To create a custom plant, basically the :py:class:`Plant` class has to
be subtyped. As the plant models the environment, it has to compute
a reward and state from sensor measurements. Together, they encode the
abstract learning target, a problem designer has in mind. The
implementation of a plant is quite straight-forward. The two functions
:py:method:`Plant.state_input` and :py:method:`Plant.reward` are called
whenever the state or reward is requested. They are expected to return
a vector (:math:`N \\times 1`) and scalar, respectively. The state space
dimension :math:`N` may be announced through the plant's constructor and
later queried by calling :py:method:`Plant.state_space_dim`. If the
plant is dependent on the episode, the :py:method:`reset` method can be
implemented as well to reset the instance's internal state. Note that
the sensor values are not preprocessed, specifically not normalized.
To do so, a normalization instance (:py:class:`PuPy.Normalization`) is 
automatically registered at :py:member:`Plant.normalization`. Note that
normalization is mandatory for :py:method:`Plant.input_state`.

"""
from plants_puppy import *
#from plants_epuck import *
