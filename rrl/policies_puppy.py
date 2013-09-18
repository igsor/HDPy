"""
ACD policies based on manipulating gait parameters. There are four
parameters and four legs; Each parameter combination (one to four) may
be controlled. Furthermore, legs may be grouped (All together,
Front/Rear, Left/Right) or individually controlled.

The class naming scheme follows:

[<Group>][<Param>]{1,4}

with <Group>:
AA = All together
FR = Front/Rear
LR = Left/Right
II = Individually

and <Param>:
A = Amplitude
P = Phase
F = Frequency
O = Offset

If several parameters are controlled at once, the <Param> identifier is
repeated. The order is A, P, F, O

"""
from rl import Policy
import numpy as np

## Base class

class GaitPolicy(Policy):
    """A concrete implementation of the :py:class:`Policy`. This one
    updates the *amplitude* parameter of a :py:class:`PuPy.Gait`.
    """
    def __init__(self, gait, action_space_dim=None):
        super(GaitPolicy, self).__init__(action_space_dim)
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

class AA1(GaitPolicy):
    """All legs; One parameter"""
    def __init__(self, gait, param_name):
        super(AA1, self).__init__(gait, 1)
        self.param = param_name
    
    def initial_action(self):
        value = self.gait.params[self.param][0]
        return np.atleast_2d([value]).T
    
    def update(self, action):
        self.gait.params[self.param] = (action[0, 0], action[0, 0], action[0, 0], action[0, 0])

class AA2(GaitPolicy):
    """All legs; Two parameters"""
    def __init__(self, gait, param_name_a, param_name_b):
        assert param_name_a != param_name_b
        super(AA2, self).__init__(gait, 2)
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

class AA3(GaitPolicy):
    """All legs; Three parameters"""
    def __init__(self, gait, param_name_a, param_name_b, param_name_c):
        assert param_name_a != param_name_b
        assert param_name_a != param_name_c
        assert param_name_b != param_name_c
        super(AA3, self).__init__(gait, 3)
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

class AA4(GaitPolicy):
    """All legs; Three parameters"""
    def __init__(self, gait):
        super(AA4, self).__init__(gait, 4)
    
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

class AAA(AA1):
    """All legs; Amplitude"""
    def __init__(self, gait):
        super(AAA, self).__init__(gait, 'amplitude')

class AAP(AA1):
    """All legs; Phase"""
    def __init__(self, gait):
        super(AAP, self).__init__(gait, 'phase')

class AAF(AA1):
    """All legs; Frequency"""
    def __init__(self, gait):
        super(AAF, self).__init__(gait, 'frequency')

class AAO(AA1):
    """All legs; Offset"""
    def __init__(self, gait):
        super(AAO, self).__init__(gait, 'offset')

# All legs, Two

class AAAP(AA2):
    """All legs; Amplitude, Phase"""
    def __init__(self, gait):
        super(AAAP, self).__init__(gait, 'amplitude', 'phase')

class AAAF(AA2):
    """All legs; Amplitude, Frequency"""
    def __init__(self, gait):
        super(AAAF, self).__init__(gait, 'amplitude', 'frequency')

class AAAO(AA2):
    """All legs; Amplitude, Offset"""
    def __init__(self, gait):
        super(AAAO, self).__init__(gait, 'amplitude', 'offset')

class AAPF(AA2):
    """All legs; Phase, Frequency"""
    def __init__(self, gait):
        super(AAPF, self).__init__(gait, 'phase', 'frequency')

class AAPO(AA2):
    """All legs; Phase, Offset"""
    def __init__(self, gait):
        super(AAPO, self).__init__(gait, 'phase', 'offset')

class AAFO(AA2):
    """All legs; Frequency, Offset"""
    def __init__(self, gait):
        super(AAFO, self).__init__(gait, 'frequency', 'offset')

# All legs, Three

class AAAPF(AA3):
    """All legs; Amplitude, Phase, Frequency"""
    def __init__(self, gait):
        super(AAAPF, self).__init__(gait, 'amplitude', 'phase', 'frequency')

class AAAPO(AA3):
    """All legs; Amplitude, Phase, Offset"""
    def __init__(self, gait):
        super(AAAPO, self).__init__(gait, 'amplitude', 'phase', 'offset')

class AAAFO(AA3):
    """All legs; Amplitude, Frequency, Offset"""
    def __init__(self, gait):
        super(AAAFO, self).__init__(gait, 'amplitude', 'frequency', 'offset')

class AAPFO(AA3):
    """All legs; Phase, Frequency, Offset"""
    def __init__(self, gait):
        super(AAPFO, self).__init__(gait, 'phase', 'frequency', 'offset')

# All legs, Four

class AAAPFO(AA4):
    """All legs; Amplitude, Phase, Frequency, Offset"""
    pass




## Front/Rear

class FR1(GaitPolicy):
    """Front/Rear; One parameter"""
    def __init__(self, gait, param_name):
        super(FR1, self).__init__(gait, 2)
        self.param = param_name
    
    def initial_action(self):
        front = self.gait.params[self.param][0]
        rear = self.gait.params[self.param][2]
        return np.atleast_2d([front, rear]).T
    
    def update(self, action):
        self.gait.params[self.param] = (action[0, 0], action[0, 0], action[1, 0], action[1, 0])

class FR2(GaitPolicy):
    """Front/Rear; Two parameters"""
    def __init__(self, gait, param_name_a, param_name_b):
        assert param_name_a != param_name_b
        super(FR2, self).__init__(gait, 4)
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

class FR3(GaitPolicy):
    """Front/Rear; Three parameters"""
    def __init__(self, gait, param_name_a, param_name_b, param_name_c):
        assert param_name_a != param_name_b
        assert param_name_a != param_name_c
        assert param_name_b != param_name_c
        super(FR3, self).__init__(gait, 6)
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

class FR4(GaitPolicy):
    """Front/Rear; Three parameters"""
    def __init__(self, gait):
        super(FR4, self).__init__(gait, 8)
    
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

class FRA(FR1):
    """Front/Rear; Amplitude"""
    def __init__(self, gait):
        super(FRA, self).__init__(gait, 'amplitude')

class FRP(FR1):
    """Front/Rear; Phase"""
    def __init__(self, gait):
        super(FRP, self).__init__(gait, 'phase')

class FRF(FR1):
    """Front/Rear; Frequency"""
    def __init__(self, gait):
        super(FRF, self).__init__(gait, 'frequency')

class FRO(FR1):
    """Front/Rear; Offset"""
    def __init__(self, gait):
        super(FRO, self).__init__(gait, 'offset')

# Front/Rear, Two

class FRAP(FR2):
    """Front/Rear; Amplitude, Phase"""
    def __init__(self, gait):
        super(FRAP, self).__init__(gait, 'amplitude', 'phase')

class FRAF(FR2):
    """Front/Rear; Amplitude, Frequency"""
    def __init__(self, gait):
        super(FRAF, self).__init__(gait, 'amplitude', 'frequency')

class FRAO(FR2):
    """Front/Rear; Amplitude, Offset"""
    def __init__(self, gait):
        super(FRAO, self).__init__(gait, 'amplitude', 'offset')

class FRPF(FR2):
    """Front/Rear; Phase, Frequency"""
    def __init__(self, gait):
        super(FRPF, self).__init__(gait, 'phase', 'frequency')

class FRPO(FR2):
    """Front/Rear; Phase, Offset"""
    def __init__(self, gait):
        super(FRPO, self).__init__(gait, 'phase', 'offset')

class FRFO(FR2):
    """Front/Rear; Frequency, Offset"""
    def __init__(self, gait):
        super(FRFO, self).__init__(gait, 'frequency', 'offset')

# Front/Rear, Three

class FRAPF(FR3):
    """Front/Rear; Amplitude, Phase, Frequency"""
    def __init__(self, gait):
        super(FRAPF, self).__init__(gait, 'amplitude', 'phase', 'frequency')

class FRAPO(FR3):
    """Front/Rear; Amplitude, Phase, Offset"""
    def __init__(self, gait):
        super(FRAPO, self).__init__(gait, 'amplitude', 'phase', 'offset')

class FRAFO(FR3):
    """Front/Rear; Amplitude, Frequency, Offset"""
    def __init__(self, gait):
        super(FRAFO, self).__init__(gait, 'amplitude', 'frequency', 'offset')

class FRPFO(FR3):
    """Front/Rear; Phase, Frequency, Offset"""
    def __init__(self, gait):
        super(FRPFO, self).__init__(gait, 'phase', 'frequency', 'offset')

# Front/Rear, Four

class FRAPFO(FR4):
    """Front/Rear; Amplitude, Phase, Frequency, Offset"""
    pass




## Left/Right

class LR1(GaitPolicy):
    """Left/Right; One parameter"""
    def __init__(self, gait, param_name):
        super(LR1, self).__init__(gait, 2)
        self.param = param_name
    
    def initial_action(self):
        left = self.gait.params[self.param][0]
        right = self.gait.params[self.param][1]
        return np.atleast_2d([left, right]).T
    
    def update(self, action):
        self.gait.params[self.param] = (action[0, 0], action[1, 0], action[0, 0], action[1, 0])

class LR2(GaitPolicy):
    """Left/Right; Two parameters"""
    def __init__(self, gait, param_name_a, param_name_b):
        assert param_name_a != param_name_b
        super(LR2, self).__init__(gait, 4)
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

class LR3(GaitPolicy):
    """Left/Right; Three parameters"""
    def __init__(self, gait, param_name_a, param_name_b, param_name_c):
        assert param_name_a != param_name_b
        assert param_name_a != param_name_c
        assert param_name_b != param_name_c
        super(LR3, self).__init__(gait, 6)
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

class LR4(GaitPolicy):
    """Front/Rear; Three parameters"""
    def __init__(self, gait):
        super(LR4, self).__init__(gait, 8)
    
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

class LRA(LR1):
    """Left/Right; Amplitude"""
    def __init__(self, gait):
        super(LRA, self).__init__(gait, 'amplitude')

class LRP(LR1):
    """Left/Right; Phase"""
    def __init__(self, gait):
        super(LRP, self).__init__(gait, 'phase')

class LRF(LR1):
    """Left/Right; Frequency"""
    def __init__(self, gait):
        super(LRF, self).__init__(gait, 'frequency')

class LRO(LR1):
    """Left/Right; Offset"""
    def __init__(self, gait):
        super(LRO, self).__init__(gait, 'offset')

# Left/Right, Two

class LRAP(LR2):
    """Left/Right; Amplitude, Phase"""
    def __init__(self, gait):
        super(LRAP, self).__init__(gait, 'amplitude', 'phase')

class LRAF(LR2):
    """Left/Right; Amplitude, Frequency"""
    def __init__(self, gait):
        super(LRAF, self).__init__(gait, 'amplitude', 'frequency')

class LRAO(LR2):
    """Left/Right; Amplitude, Offset"""
    def __init__(self, gait):
        super(LRAO, self).__init__(gait, 'amplitude', 'offset')

class LRPF(LR2):
    """Left/Right; Phase, Frequency"""
    def __init__(self, gait):
        super(LRPF, self).__init__(gait, 'phase', 'frequency')

class LRPO(LR2):
    """Left/Right; Phase, Offset"""
    def __init__(self, gait):
        super(LRPO, self).__init__(gait, 'phase', 'offset')

class LRFO(LR2):
    """Left/Right; Frequency, Offset"""
    def __init__(self, gait):
        super(LRFO, self).__init__(gait, 'frequency', 'offset')

# Left/Right, Three

class LRAPF(LR3):
    """Left/Right; Amplitude, Phase, Frequency"""
    def __init__(self, gait):
        super(LRAPF, self).__init__(gait, 'amplitude', 'phase', 'frequency')

class LRAPO(LR3):
    """Left/Right; Amplitude, Phase, Offset"""
    def __init__(self, gait):
        super(LRAPO, self).__init__(gait, 'amplitude', 'phase', 'offset')

class LRAFO(LR3):
    """Left/Right; Amplitude, Frequency, Offset"""
    def __init__(self, gait):
        super(LRAFO, self).__init__(gait, 'amplitude', 'frequency', 'offset')

class LRPFO(LR3):
    """Left/Right; Phase, Frequency, Offset"""
    def __init__(self, gait):
        super(LRPFO, self).__init__(gait, 'phase', 'frequency', 'offset')

# Left/Right, Four

class LRAPFO(LR4):
    """Left/Right; Amplitude, Phase, Frequency, Offset"""
    pass




## Individual

class II1(GaitPolicy):
    """All legs; One parameter"""
    def __init__(self, gait, param_name):
        super(II1, self).__init__(gait, 4)
        self.param = param_name
    
    def initial_action(self):
        return np.atleast_2d(self.gait.params[self.param]).T
    
    def update(self, action):
        self.gait.params[self.param] = (action[0, 0], action[1, 0], action[2, 0], action[3, 0])

class II2(GaitPolicy):
    """All legs; Two parameters"""
    def __init__(self, gait, param_name_a, param_name_b):
        assert param_name_a != param_name_b
        super(II2, self).__init__(gait, 8)
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

class II3(GaitPolicy):
    """All legs; Three parameters"""
    def __init__(self, gait, param_name_a, param_name_b, param_name_c):
        assert param_name_a != param_name_b
        assert param_name_a != param_name_c
        assert param_name_b != param_name_c
        super(II3, self).__init__(gait, 12)
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

class II4(GaitPolicy):
    """All legs; Three parameters"""
    def __init__(self, gait):
        super(II4, self).__init__(gait, 16)
    
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

class IIA(II1):
    """Individual; Amplitude"""
    def __init__(self, gait):
        super(IIA, self).__init__(gait, 'amplitude')

class IIP(II1):
    """Individual; Phase"""
    def __init__(self, gait):
        super(IIP, self).__init__(gait, 'phase')

class IIF(II1):
    """Individual; Frequency"""
    def __init__(self, gait):
        super(IIF, self).__init__(gait, 'frequency')

class IIO(II1):
    """Individual; Offset"""
    def __init__(self, gait):
        super(IIO, self).__init__(gait, 'offset')

# Individual, Two

class IIAP(II2):
    """Individual; Amplitude, Phase"""
    def __init__(self, gait):
        super(IIAP, self).__init__(gait, 'amplitude', 'phase')

class IIAF(II2):
    """Individual; Amplitude, Frequency"""
    def __init__(self, gait):
        super(IIAF, self).__init__(gait, 'amplitude', 'frequency')

class IIAO(II2):
    """Individual; Amplitude, Offset"""
    def __init__(self, gait):
        super(IIAO, self).__init__(gait, 'amplitude', 'offset')

class IIPF(II2):
    """Individual; Phase, Frequency"""
    def __init__(self, gait):
        super(IIPF, self).__init__(gait, 'phase', 'frequency')

class IIPO(II2):
    """Individual; Phase, Offset"""
    def __init__(self, gait):
        super(IIPO, self).__init__(gait, 'phase', 'offset')

class IIFO(II2):
    """Individual; Frequency, Offset"""
    def __init__(self, gait):
        super(IIFO, self).__init__(gait, 'frequency', 'offset')

# Individual, Three

class IIAPF(II3):
    """Individual; Amplitude, Phase, Frequency"""
    def __init__(self, gait):
        super(IIAPF, self).__init__(gait, 'amplitude', 'phase', 'frequency')

class IIAPO(II3):
    """Individual; Amplitude, Phase, Offset"""
    def __init__(self, gait):
        super(IIAPO, self).__init__(gait, 'amplitude', 'phase', 'offset')

class IIAFO(II3):
    """Individual; Amplitude, Frequency, Offset"""
    def __init__(self, gait):
        super(IIAFO, self).__init__(gait, 'amplitude', 'frequency', 'offset')

class IIPFO(II3):
    """Individual; Phase, Frequency, Offset"""
    def __init__(self, gait):
        super(IIPFO, self).__init__(gait, 'phase', 'frequency', 'offset')

# Individual, Four

class IIAPFO(II4):
    """Individual; Amplitude, Phase, Frequency, Offset"""
    pass



