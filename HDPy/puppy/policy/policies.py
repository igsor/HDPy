"""
Puppy's policies are based on manipulating gait parameters. There
are four parameters and four legs, hence a total of 16 parameters. The
parameters under control may be reduced (one to four) and legs may be
grouped (All together, Front/Rear, Left/Right).

The class naming scheme is as follows:

[<Group>][<Param>]{1,4}

with <Group>:

+----+--------------+
| AA | All together |
+----+--------------+
| FR | Front/Rear   |
+----+--------------+
| LR | Left/Right   |
+----+--------------+
| II | Individually |
+----+--------------+


and <Param>:

+---+-----------+
| A | Amplitude |
+---+-----------+
| P | Phase     |
+---+-----------+
| F | Frequency |
+---+-----------+
| O | Offset    |
+---+-----------+

If several parameters are controlled at once, the <Param> identifier is
repeated up to four times. The order is A, P, F, O.

"""
from ...rl import Policy
import numpy as np

## Base class

class _GaitPolicy(Policy):
    """A concrete implementation of the :py:class:`Policy`. This one
    updates the *amplitude* parameter of a :py:class:`PuPy.Gait`.
    """
    def __init__(self, gait, action_space_dim=None):
        super(_GaitPolicy, self).__init__(action_space_dim)
        self.gait = gait
        self.gait_orig = gait.copy()
    
    def get_iterator(self, time_start_ms, time_end_ms, step_size_ms):
        """Return an iterator for the *motor_target* sequence.
        """
        return self.gait.iter(time_start_ms, step_size_ms)
    
    def reset(self):
        """Reset the gait to the original values."""
        self.gait = self.gait_orig.copy()
    
    def update(self, action):
        raise NotImplementedError()
    
    def initial_action(self):
        raise NotImplementedError()

## All legs

class _AA1(_GaitPolicy):
    """All legs; One parameter"""
    def __init__(self, gait, param_name):
        super(_AA1, self).__init__(gait, 1)
        self.param = param_name
    
    def initial_action(self):
        value = self.gait.params[self.param][0]
        return np.atleast_2d([value]).T
    
    def update(self, action):
        self.gait.params[self.param] = (action[0, 0], action[0, 0], action[0, 0], action[0, 0])

class _AA2(_GaitPolicy):
    """All legs; Two parameters"""
    def __init__(self, gait, param_name_a, param_name_b):
        assert param_name_a != param_name_b
        super(_AA2, self).__init__(gait, 2)
        self.params = (param_name_a, param_name_b)
    
    def initial_action(self):
        param_0, param_1 = self.params
        value_0 = self.gait.params[param_0][0]
        value_1 = self.gait.params[param_1][0]
        return np.atleast_2d([value_0, value_1]).T
    
    def update(self, action):
        param_0, param_1 = self.params
        self.gait.params[param_0] = (action[0, 0], action[0, 0], action[0, 0], action[0, 0])
        self.gait.params[param_1] = (action[1, 0], action[1, 0], action[1, 0], action[1, 0])

class _AA3(_GaitPolicy):
    """All legs; Three parameters"""
    def __init__(self, gait, param_name_a, param_name_b, param_name_c):
        assert param_name_a != param_name_b
        assert param_name_a != param_name_c
        assert param_name_b != param_name_c
        super(_AA3, self).__init__(gait, 3)
        self.params = (param_name_a, param_name_b, param_name_c)
    
    def initial_action(self):
        param_0, param_1, param_2 = self.params
        value_0 = self.gait.params[param_0][0]
        value_1 = self.gait.params[param_1][0]
        value_2 = self.gait.params[param_2][0]
        return np.atleast_2d([value_0, value_1, value_2]).T
    
    def update(self, action):
        param_0, param_1, param_2 = self.params
        self.gait.params[param_0] = (action[0, 0], action[0, 0], action[0, 0], action[0, 0])
        self.gait.params[param_1] = (action[1, 0], action[1, 0], action[1, 0], action[1, 0])
        self.gait.params[param_2] = (action[2, 0], action[2, 0], action[2, 0], action[2, 0])

class _AA4(_GaitPolicy):
    """All legs; Three parameters"""
    def __init__(self, gait):
        super(_AA4, self).__init__(gait, 4)
    
    def initial_action(self):
        params = ('amplitude', 'frequency', 'offset', 'phase')
        vals = []
        for i in params:
            vals.append(self.gait.params[i][0])
        
        return np.atleast_2d(vals).T
    
    def update(self, action):
        self.gait.params['amplitude']   = (action[0, 0], action[0, 0], action[0, 0], action[0, 0])
        self.gait.params['frequency']   = (action[1, 0], action[1, 0], action[1, 0], action[1, 0])
        self.gait.params['offset']      = (action[2, 0], action[2, 0], action[2, 0], action[2, 0])
        self.gait.params['phase']       = (action[3, 0], action[3, 0], action[3, 0], action[3, 0])

# All legs, One

class AAA(_AA1):
    """All legs; Amplitude"""
    def __init__(self, gait):
        super(AAA, self).__init__(gait, 'amplitude')

class AAP(_AA1):
    """All legs; Phase"""
    def __init__(self, gait):
        super(AAP, self).__init__(gait, 'phase')

class AAF(_AA1):
    """All legs; Frequency"""
    def __init__(self, gait):
        super(AAF, self).__init__(gait, 'frequency')

class AAO(_AA1):
    """All legs; Offset"""
    def __init__(self, gait):
        super(AAO, self).__init__(gait, 'offset')

# All legs, Two

class AAAP(_AA2):
    """All legs; Amplitude, Phase"""
    def __init__(self, gait):
        super(AAAP, self).__init__(gait, 'amplitude', 'phase')

class AAAF(_AA2):
    """All legs; Amplitude, Frequency"""
    def __init__(self, gait):
        super(AAAF, self).__init__(gait, 'amplitude', 'frequency')

class AAAO(_AA2):
    """All legs; Amplitude, Offset"""
    def __init__(self, gait):
        super(AAAO, self).__init__(gait, 'amplitude', 'offset')

class AAPF(_AA2):
    """All legs; Phase, Frequency"""
    def __init__(self, gait):
        super(AAPF, self).__init__(gait, 'phase', 'frequency')

class AAPO(_AA2):
    """All legs; Phase, Offset"""
    def __init__(self, gait):
        super(AAPO, self).__init__(gait, 'phase', 'offset')

class AAFO(_AA2):
    """All legs; Frequency, Offset"""
    def __init__(self, gait):
        super(AAFO, self).__init__(gait, 'frequency', 'offset')

# All legs, Three

class AAAPF(_AA3):
    """All legs; Amplitude, Phase, Frequency"""
    def __init__(self, gait):
        super(AAAPF, self).__init__(gait, 'amplitude', 'phase', 'frequency')

class AAAPO(_AA3):
    """All legs; Amplitude, Phase, Offset"""
    def __init__(self, gait):
        super(AAAPO, self).__init__(gait, 'amplitude', 'phase', 'offset')

class AAAFO(_AA3):
    """All legs; Amplitude, Frequency, Offset"""
    def __init__(self, gait):
        super(AAAFO, self).__init__(gait, 'amplitude', 'frequency', 'offset')

class AAPFO(_AA3):
    """All legs; Phase, Frequency, Offset"""
    def __init__(self, gait):
        super(AAPFO, self).__init__(gait, 'phase', 'frequency', 'offset')

# All legs, Four

class AAAPFO(_AA4):
    """All legs; Amplitude, Phase, Frequency, Offset"""
    pass




## Front/Rear

class _FR1(_GaitPolicy):
    """Front/Rear; One parameter"""
    def __init__(self, gait, param_name):
        super(_FR1, self).__init__(gait, 2)
        self.param = param_name
    
    def initial_action(self):
        front = self.gait.params[self.param][0]
        rear = self.gait.params[self.param][2]
        return np.atleast_2d([front, rear]).T
    
    def update(self, action):
        self.gait.params[self.param] = (action[0, 0], action[0, 0], action[1, 0], action[1, 0])

class _FR2(_GaitPolicy):
    """Front/Rear; Two parameters"""
    def __init__(self, gait, param_name_a, param_name_b):
        assert param_name_a != param_name_b
        super(_FR2, self).__init__(gait, 4)
        self.params = (param_name_a, param_name_b)
    
    def initial_action(self):
        param_0, param_1 = self.params
        
        front_0 = self.gait.params[param_0][0]
        rear_0 = self.gait.params[param_0][2]
        
        front_1 = self.gait.params[param_1][0]
        rear_1 = self.gait.params[param_1][2]
        
        return np.atleast_2d([front_0, rear_0, front_1, rear_1]).T
    
    def update(self, action):
        param_0, param_1 = self.params
        self.gait.params[param_0] = (action[0, 0], action[0, 0], action[1, 0], action[1, 0])
        self.gait.params[param_1] = (action[2, 0], action[2, 0], action[3, 0], action[3, 0])

class _FR3(_GaitPolicy):
    """Front/Rear; Three parameters"""
    def __init__(self, gait, param_name_a, param_name_b, param_name_c):
        assert param_name_a != param_name_b
        assert param_name_a != param_name_c
        assert param_name_b != param_name_c
        super(_FR3, self).__init__(gait, 6)
        self.params = (param_name_a, param_name_b, param_name_c)
    
    def initial_action(self):
        param_0, param_1, param_2 = self.params
        
        front_0 = self.gait.params[param_0][0]
        rear_0 = self.gait.params[param_0][2]
        
        front_1 = self.gait.params[param_1][0]
        rear_1 = self.gait.params[param_1][2]
        
        front_2 = self.gait.params[param_2][0]
        rear_2 = self.gait.params[param_2][2]
        
        return np.atleast_2d([front_0, rear_0, front_1, rear_1, front_2, rear_2]).T
    
    def update(self, action):
        param_0, param_1, param_2 = self.params
        self.gait.params[param_0] = (action[0, 0], action[0, 0], action[1, 0], action[1, 0])
        self.gait.params[param_1] = (action[2, 0], action[2, 0], action[3, 0], action[3, 0])
        self.gait.params[param_2] = (action[4, 0], action[4, 0], action[5, 0], action[5, 0])

class _FR4(_GaitPolicy):
    """Front/Rear; Three parameters"""
    def __init__(self, gait):
        super(_FR4, self).__init__(gait, 8)
    
    def initial_action(self):
        params = ('amplitude', 'frequency', 'offset', 'phase')
        vals = []
        for i in params:
            vals.append(self.gait.params[i][0])
            vals.append(self.gait.params[i][2])
        
        return np.atleast_2d(vals).T
    
    def update(self, action):
        self.gait.params['amplitude']   = (action[0, 0], action[0, 0], action[1, 0], action[1, 0])
        self.gait.params['frequency']   = (action[2, 0], action[2, 0], action[3, 0], action[3, 0])
        self.gait.params['offset']      = (action[4, 0], action[4, 0], action[5, 0], action[5, 0])
        self.gait.params['phase']       = (action[6, 0], action[6, 0], action[7, 0], action[7, 0])

# Front/Rear, One

class FRA(_FR1):
    """Front/Rear; Amplitude"""
    def __init__(self, gait):
        super(FRA, self).__init__(gait, 'amplitude')

class FRP(_FR1):
    """Front/Rear; Phase"""
    def __init__(self, gait):
        super(FRP, self).__init__(gait, 'phase')

class FRF(_FR1):
    """Front/Rear; Frequency"""
    def __init__(self, gait):
        super(FRF, self).__init__(gait, 'frequency')

class FRO(_FR1):
    """Front/Rear; Offset"""
    def __init__(self, gait):
        super(FRO, self).__init__(gait, 'offset')

# Front/Rear, Two

class FRAP(_FR2):
    """Front/Rear; Amplitude, Phase"""
    def __init__(self, gait):
        super(FRAP, self).__init__(gait, 'amplitude', 'phase')

class FRAF(_FR2):
    """Front/Rear; Amplitude, Frequency"""
    def __init__(self, gait):
        super(FRAF, self).__init__(gait, 'amplitude', 'frequency')

class FRAO(_FR2):
    """Front/Rear; Amplitude, Offset"""
    def __init__(self, gait):
        super(FRAO, self).__init__(gait, 'amplitude', 'offset')

class FRPF(_FR2):
    """Front/Rear; Phase, Frequency"""
    def __init__(self, gait):
        super(FRPF, self).__init__(gait, 'phase', 'frequency')

class FRPO(_FR2):
    """Front/Rear; Phase, Offset"""
    def __init__(self, gait):
        super(FRPO, self).__init__(gait, 'phase', 'offset')

class FRFO(_FR2):
    """Front/Rear; Frequency, Offset"""
    def __init__(self, gait):
        super(FRFO, self).__init__(gait, 'frequency', 'offset')

# Front/Rear, Three

class FRAPF(_FR3):
    """Front/Rear; Amplitude, Phase, Frequency"""
    def __init__(self, gait):
        super(FRAPF, self).__init__(gait, 'amplitude', 'phase', 'frequency')

class FRAPO(_FR3):
    """Front/Rear; Amplitude, Phase, Offset"""
    def __init__(self, gait):
        super(FRAPO, self).__init__(gait, 'amplitude', 'phase', 'offset')

class FRAFO(_FR3):
    """Front/Rear; Amplitude, Frequency, Offset"""
    def __init__(self, gait):
        super(FRAFO, self).__init__(gait, 'amplitude', 'frequency', 'offset')

class FRPFO(_FR3):
    """Front/Rear; Phase, Frequency, Offset"""
    def __init__(self, gait):
        super(FRPFO, self).__init__(gait, 'phase', 'frequency', 'offset')

# Front/Rear, Four

class FRAPFO(_FR4):
    """Front/Rear; Amplitude, Phase, Frequency, Offset"""
    pass




## Left/Right

class _LR1(_GaitPolicy):
    """Left/Right; One parameter"""
    def __init__(self, gait, param_name):
        super(_LR1, self).__init__(gait, 2)
        self.param = param_name
    
    def initial_action(self):
        left = self.gait.params[self.param][0]
        right = self.gait.params[self.param][1]
        return np.atleast_2d([left, right]).T
    
    def update(self, action):
        self.gait.params[self.param] = (action[0, 0], action[1, 0], action[0, 0], action[1, 0])

class _LR2(_GaitPolicy):
    """Left/Right; Two parameters"""
    def __init__(self, gait, param_name_a, param_name_b):
        assert param_name_a != param_name_b
        super(_LR2, self).__init__(gait, 4)
        self.params = (param_name_a, param_name_b)
    
    def initial_action(self):
        param_0, param_1 = self.params
        
        left_0 = self.gait.params[param_0][0]
        right_0 = self.gait.params[param_0][1]
        
        left_1 = self.gait.params[param_1][0]
        right_1 = self.gait.params[param_1][1]
        
        return np.atleast_2d([left_0, right_0, left_1, right_1]).T
    
    def update(self, action):
        param_0, param_1 = self.params
        self.gait.params[param_0] = (action[0, 0], action[1, 0], action[0, 0], action[1, 0])
        self.gait.params[param_1] = (action[2, 0], action[3, 0], action[2, 0], action[3, 0])

class _LR3(_GaitPolicy):
    """Left/Right; Three parameters"""
    def __init__(self, gait, param_name_a, param_name_b, param_name_c):
        assert param_name_a != param_name_b
        assert param_name_a != param_name_c
        assert param_name_b != param_name_c
        super(_LR3, self).__init__(gait, 6)
        self.params = (param_name_a, param_name_b, param_name_c)
    
    def initial_action(self):
        param_0, param_1, param_2 = self.params
        
        left_0 = self.gait.params[param_0][0]
        right_0 = self.gait.params[param_0][1]
        
        left_1 = self.gait.params[param_1][0]
        right_1 = self.gait.params[param_1][1]
        
        left_2 = self.gait.params[param_2][0]
        right_2 = self.gait.params[param_2][1]
        
        return np.atleast_2d([left_0, right_0, left_1, right_1, left_2, right_2]).T
    
    def update(self, action):
        param_0, param_1, param_2 = self.params
        self.gait.params[param_0] = (action[0, 0], action[1, 0], action[0, 0], action[1, 0])
        self.gait.params[param_1] = (action[2, 0], action[3, 0], action[2, 0], action[3, 0])
        self.gait.params[param_2] = (action[4, 0], action[5, 0], action[4, 0], action[5, 0])

class _LR4(_GaitPolicy):
    """Front/Rear; Three parameters"""
    def __init__(self, gait):
        super(_LR4, self).__init__(gait, 8)
    
    def initial_action(self):
        params = ('amplitude', 'frequency', 'offset', 'phase')
        vals = []
        for i in params:
            vals.append(self.gait.params[i][0])
            vals.append(self.gait.params[i][1])
        
        return np.atleast_2d(vals).T
    
    def update(self, action):
        self.gait.params['amplitude']   = (action[0, 0], action[1, 0], action[0, 0], action[1, 0])
        self.gait.params['frequency']   = (action[2, 0], action[3, 0], action[2, 0], action[3, 0])
        self.gait.params['offset']      = (action[4, 0], action[5, 0], action[4, 0], action[5, 0])
        self.gait.params['phase']       = (action[6, 0], action[7, 0], action[6, 0], action[7, 0])

# Left/Right, One

class LRA(_LR1):
    """Left/Right; Amplitude"""
    def __init__(self, gait):
        super(LRA, self).__init__(gait, 'amplitude')

class LRP(_LR1):
    """Left/Right; Phase"""
    def __init__(self, gait):
        super(LRP, self).__init__(gait, 'phase')

class LRF(_LR1):
    """Left/Right; Frequency"""
    def __init__(self, gait):
        super(LRF, self).__init__(gait, 'frequency')

class LRO(_LR1):
    """Left/Right; Offset"""
    def __init__(self, gait):
        super(LRO, self).__init__(gait, 'offset')

# Left/Right, Two

class LRAP(_LR2):
    """Left/Right; Amplitude, Phase"""
    def __init__(self, gait):
        super(LRAP, self).__init__(gait, 'amplitude', 'phase')

class LRAF(_LR2):
    """Left/Right; Amplitude, Frequency"""
    def __init__(self, gait):
        super(LRAF, self).__init__(gait, 'amplitude', 'frequency')

class LRAO(_LR2):
    """Left/Right; Amplitude, Offset"""
    def __init__(self, gait):
        super(LRAO, self).__init__(gait, 'amplitude', 'offset')

class LRPF(_LR2):
    """Left/Right; Phase, Frequency"""
    def __init__(self, gait):
        super(LRPF, self).__init__(gait, 'phase', 'frequency')

class LRPO(_LR2):
    """Left/Right; Phase, Offset"""
    def __init__(self, gait):
        super(LRPO, self).__init__(gait, 'phase', 'offset')

class LRFO(_LR2):
    """Left/Right; Frequency, Offset"""
    def __init__(self, gait):
        super(LRFO, self).__init__(gait, 'frequency', 'offset')

# Left/Right, Three

class LRAPF(_LR3):
    """Left/Right; Amplitude, Phase, Frequency"""
    def __init__(self, gait):
        super(LRAPF, self).__init__(gait, 'amplitude', 'phase', 'frequency')

class LRAPO(_LR3):
    """Left/Right; Amplitude, Phase, Offset"""
    def __init__(self, gait):
        super(LRAPO, self).__init__(gait, 'amplitude', 'phase', 'offset')

class LRAFO(_LR3):
    """Left/Right; Amplitude, Frequency, Offset"""
    def __init__(self, gait):
        super(LRAFO, self).__init__(gait, 'amplitude', 'frequency', 'offset')

class LRPFO(_LR3):
    """Left/Right; Phase, Frequency, Offset"""
    def __init__(self, gait):
        super(LRPFO, self).__init__(gait, 'phase', 'frequency', 'offset')

# Left/Right, Four

class LRAPFO(_LR4):
    """Left/Right; Amplitude, Phase, Frequency, Offset"""
    pass




## Individual

class _II1(_GaitPolicy):
    """All legs; One parameter"""
    def __init__(self, gait, param_name):
        super(_II1, self).__init__(gait, 4)
        self.param = param_name
    
    def initial_action(self):
        return np.atleast_2d(self.gait.params[self.param]).T
    
    def update(self, action):
        self.gait.params[self.param] = (action[0, 0], action[1, 0], action[2, 0], action[3, 0])

class _II2(_GaitPolicy):
    """All legs; Two parameters"""
    def __init__(self, gait, param_name_a, param_name_b):
        assert param_name_a != param_name_b
        super(_II2, self).__init__(gait, 8)
        self.params = (param_name_a, param_name_b)
    
    def initial_action(self):
        param_0, param_1 = self.params
        value_0 = self.gait.params[param_0]
        value_1 = self.gait.params[param_1]
        return np.atleast_2d(value_0 + value_1).T
    
    def update(self, action):
        param_0, param_1 = self.params
        self.gait.params[param_0] = (action[0, 0], action[1, 0], action[2, 0], action[3, 0])
        self.gait.params[param_1] = (action[4, 0], action[5, 0], action[6, 0], action[7, 0])

class _II3(_GaitPolicy):
    """All legs; Three parameters"""
    def __init__(self, gait, param_name_a, param_name_b, param_name_c):
        assert param_name_a != param_name_b
        assert param_name_a != param_name_c
        assert param_name_b != param_name_c
        super(_II3, self).__init__(gait, 12)
        self.params = (param_name_a, param_name_b, param_name_c)
    
    def initial_action(self):
        param_0, param_1, param_2 = self.params
        value_0 = self.gait.params[param_0]
        value_1 = self.gait.params[param_1]
        value_2 = self.gait.params[param_2]
        return np.atleast_2d(value_0 + value_1 + value_2).T
    
    def update(self, action):
        param_0, param_1, param_2 = self.params
        self.gait.params[param_0] = (action[0, 0], action[1, 0], action[2, 0], action[3, 0])
        self.gait.params[param_1] = (action[4, 0], action[5, 0], action[6, 0], action[7, 0])
        self.gait.params[param_2] = (action[8, 0], action[9, 0], action[10, 0], action[11, 0])

class _II4(_GaitPolicy):
    """All legs; Three parameters"""
    def __init__(self, gait):
        super(_II4, self).__init__(gait, 16)
    
    def initial_action(self):
        params = ('amplitude', 'frequency', 'offset', 'phase')
        values = []
        for i in params:
            values.extend(self.gait.params[i])
        return np.atleast_2d(values).T
    
    def update(self, action):
        self.gait.params['amplitude']   = (action[0, 0], action[1, 0], action[2, 0], action[3, 0])
        self.gait.params['frequency']   = (action[4, 0], action[5, 0], action[6, 0], action[7, 0])
        self.gait.params['offset']      = (action[8, 0], action[9, 0], action[10, 0], action[11, 0])
        self.gait.params['phase']       = (action[12, 0], action[13, 0], action[14, 0], action[15, 0])

# Individual, One

class IIA(_II1):
    """Individual; Amplitude"""
    def __init__(self, gait):
        super(IIA, self).__init__(gait, 'amplitude')

class IIP(_II1):
    """Individual; Phase"""
    def __init__(self, gait):
        super(IIP, self).__init__(gait, 'phase')

class IIF(_II1):
    """Individual; Frequency"""
    def __init__(self, gait):
        super(IIF, self).__init__(gait, 'frequency')

class IIO(_II1):
    """Individual; Offset"""
    def __init__(self, gait):
        super(IIO, self).__init__(gait, 'offset')

# Individual, Two

class IIAP(_II2):
    """Individual; Amplitude, Phase"""
    def __init__(self, gait):
        super(IIAP, self).__init__(gait, 'amplitude', 'phase')

class IIAF(_II2):
    """Individual; Amplitude, Frequency"""
    def __init__(self, gait):
        super(IIAF, self).__init__(gait, 'amplitude', 'frequency')

class IIAO(_II2):
    """Individual; Amplitude, Offset"""
    def __init__(self, gait):
        super(IIAO, self).__init__(gait, 'amplitude', 'offset')

class IIPF(_II2):
    """Individual; Phase, Frequency"""
    def __init__(self, gait):
        super(IIPF, self).__init__(gait, 'phase', 'frequency')

class IIPO(_II2):
    """Individual; Phase, Offset"""
    def __init__(self, gait):
        super(IIPO, self).__init__(gait, 'phase', 'offset')

class IIFO(_II2):
    """Individual; Frequency, Offset"""
    def __init__(self, gait):
        super(IIFO, self).__init__(gait, 'frequency', 'offset')

# Individual, Three

class IIAPF(_II3):
    """Individual; Amplitude, Phase, Frequency"""
    def __init__(self, gait):
        super(IIAPF, self).__init__(gait, 'amplitude', 'phase', 'frequency')

class IIAPO(_II3):
    """Individual; Amplitude, Phase, Offset"""
    def __init__(self, gait):
        super(IIAPO, self).__init__(gait, 'amplitude', 'phase', 'offset')

class IIAFO(_II3):
    """Individual; Amplitude, Frequency, Offset"""
    def __init__(self, gait):
        super(IIAFO, self).__init__(gait, 'amplitude', 'frequency', 'offset')

class IIPFO(_II3):
    """Individual; Phase, Frequency, Offset"""
    def __init__(self, gait):
        super(IIPFO, self).__init__(gait, 'phase', 'frequency', 'offset')

# Individual, Four

class IIAPFO(_II4):
    """Individual; Amplitude, Phase, Frequency, Offset"""
    pass

# Policy with raw motor targets
class RawTrgPolicy(Policy):
    """not yet working well..."""
    def __init__(self, init_trg=np.atleast_2d([0.0, 0.0, 0.0, 0.0]).T, init_gait=None):
        super(RawTrgPolicy, self).__init__(4)
        self.init_gait = init_gait
#        if self.init_gait is not None:
#            self.init=
        self.init_trg = init_trg
        self.trg = init_trg
    
    def get_iterator(self, time_start_ms, time_end_ms, step_size_ms):
        """Return an iterator for the *motor_target* sequence.
        """
        def _iter():
            while True:
                yield list(self.trg[:,0])
#        _print("call get_iterator "+str(time_start_ms))
        return _iter()
    
    def reset(self):
        self.trg = self.init_trg.copy()
#        _print("call reset")
    
    def update(self, action):
        self.trg = action
#        _print("call update")
#        print "POLICY: new action=", action.T
    
    def initial_action(self):
#        _print("call initial_action")
        return self.trg


