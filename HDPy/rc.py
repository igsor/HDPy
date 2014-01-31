"""
As Reservoir Computing is uses in this module, an implementation of an
Echo State Network is provided through a couple of functions. The
reservoir is given through :py:class:`ReservoirNode`. Note that although
the dependency has been removed, most part of the reservoir code is
copied from [Oger]_ and [MDP]_. The reservoir has then be combined with
a readout, usually a linear regression. For the offline case, a ridge
regression would probably be incorporated (available in
:py:class:`Oger.nodes.RidgeRegressionNode`). Since for this module, only
the online case is relevant, a recursive least-squares filter is
implemented in :py:class:`StabilizedRLS`. The reservoir and readout are
easily combined by feeding the reservoir state as sample into the RLS
(together with a training target).

In contrast to [Oger]_, the reservoir is set up through three external
functions. These define how the weight matrices are initialized. The
default approach is to use :py:func:`dense_w_in`, :py:func:`sparse_w_in`
and :py:func:`dense_w_bias`. However, other reservoir setup functions
are provided. Note the documentation in :py:class:`ReservoirNode` if
a different initialization has to be implemented.

To test reservoir memory, two functions are provided,
:py:func:`reservoir_memory` and :py:func:`find_radius_for_mc`. Both of
them measure the settling time of the impulse response, which is related
to the spectral radius.

"""
import warnings
import numpy as np
import scipy.sparse
import copy as _copy
import pickle as _cPickle


## RESERVOIR SETUP FUNCTIONS ##

class dense_w_in(object):
    """Dense input matrix.
    
    ``scaling``
        Input scaling.
    
    ``rnd_fu``
        Callback for random number generation. Expects the argument
        *size*. Default is a uniform distribution in [-1,1].
    
    """
    def __init__(self, scaling, rnd_fu=None):
        self._scaling = scaling
        self._rnd_fu = rnd_fu
    
    def _rnd_gen(self, **kwargs):
        if self._rnd_fu is not None:
            return self._rnd_fu(**kwargs)
        else:
            return np.random.uniform(**kwargs)*2.0-1.0
    
    def __call__(self, out_size, in_size):
        """Return a dense connection matrix for ``in_size`` inputs
        to a reservoir of ``out_size`` nodes."""
        return scipy.sparse.csc_matrix(self._scaling * self._rnd_gen(size=(out_size, in_size)))

class sparse_w_in(object):
    """Sparse, random reservoir input. The density is exactly as
    defined by ``density``, given in percent (0-100). The values are
    drawn from an uniform distribution.
    
    ``scaling``
        Scaling of input weights.
    
    ``density``
        Percentage of non-trivial connection weights, [0-100].
    
    ``rnd_fu``
        Random number generator for the weight values. Must take
        the scalar argument *size* that determines the length of
        the generated random number list. The default is the uniform
        distribution between -1.0 and 1.0.
    
    """ 
    def __init__(self, scaling, density, rnd_fu=None):
        self._scaling = scaling
        self._density = density
        self._rnd_fu = rnd_fu
    
    def _rnd_gen(self, **kwargs):
        if self._rnd_fu is not None:
            return self._rnd_fu(**kwargs)
        else:
            return np.random.uniform(**kwargs)*2.0-1.0
    
    def __call__(self, out_size, in_size):
        """Return a sparse connection matrix for ``in_size`` inputs
        to a reservoir of ``out_size`` nodes."""
        nrentries = np.int32(out_size * in_size * self._density / 100.0)
        idx = np.random.choice(out_size*in_size, nrentries, False)
        locs = np.array(zip(*map(lambda i: (i/in_size, i%in_size), idx)))
        datavec = self._rnd_gen(size=nrentries)
        w = self._scaling * scipy.sparse.csc_matrix((datavec, locs), shape=(out_size, in_size))
        return w

class dense_w_bias(object):
    """Dense reservoir node bias. The weights are picked from a random
    number generator ``rnd_fu`` and scaled according to ``scaling``.
    
    ``scaling``
        Scale of the bias weights
    
    ``rnd_fu``
        Random number generator for the weights. Default is normal
        distribution in [-1.0, 1.0].
    """
    def __init__(self, scaling, rnd_fu=None):
        self._scaling = scaling
        self._rnd_fu = rnd_fu
    
    def _rnd_gen(self, **kwargs):
        if self._rnd_fu is not None:
            return self._rnd_fu(**kwargs)
        else:
            return np.random.normal(**kwargs)*2.0-1.0
    
    def __call__(self, out_size):
        """Return a dense bias matrix for ``out_size`` reservoir nodes."""
        return self._scaling * self._rnd_gen(size=(1, out_size))

class sparse_reservoir(object):
    """Reservoir with sparse connected neurons and random connection
    weights. The non-zero weights are uniformly distributed, the
    values drawn from a random distribution.
    
    ``density``
        Reservoir density, given in percent (0-100). 
    
    ``rnd_fu``
        Random number generator for the weight values. Must take
        the scalar argument *size* that determines the length of
        the generated random number list. The default is the normal
        distribution.
    
    """
    def __init__(self, density, rnd_fu=None):
        self._density = density
        self._rnd_fu = rnd_fu
    
    def _rnd_gen(self, **kwargs):
        if self._rnd_fu is not None:
            return self._rnd_fu(**kwargs)
        else:
            return np.random.normal(**kwargs)
    
    def __call__(self, out_size, specrad):
        """Return a sparse reservoir matrix of size ``out_size`` with
        spectral radius ``specrad``."""
        from scipy.sparse.linalg import eigs, ArpackNoConvergence
        converged = False
        # Initialize reservoir weight matrix
        nrentries = int((out_size**2 * self._density)/100.0)
        out_size = int(out_size)
        # Keep generating random matrices until convergence
        num_iter = 1000
        while not converged:
            try:
                idx = np.random.choice(out_size**2, nrentries, False)
                locs = np.array(zip(*map(lambda i: (i/out_size, i%out_size), idx)))
                datavec = self._rnd_gen(size=nrentries)
                w = scipy.sparse.csc_matrix((datavec, locs), shape=(out_size, out_size))
                eigvals = eigs(w, return_eigenvectors=False, k=3)
                converged = True
                w *= (specrad / np.amax(np.absolute(eigvals)))
            except ArpackNoConvergence:
                num_iter -= 1
                if num_iter < 1:
                    raise Exception('Reservoir matrix could not be built')
        return w

class dense_reservoir(object):
    """Reservoir with densely connected neurons and random connection
    weights.
    
    ``rnd_fu``
        Random number generator for the weight values. Must take
        the scalar argument *size* that determines the length of
        the generated random number list. The default is the normal
        distribution.
    
    """
    def __init__(self, rnd_fu=None):
        self._rnd_fu = rnd_fu
    
    def _rnd_fu(self, **kwargs):
        if self._rnd_fu is not None:
            return self._rnd_fu(**kwargs)
        else:
            return np.random.normal(**kwargs)
    
    def __call__(self, out_size, specrad):
        """Return a dense reservoir matrix of size ``out_size`` with
        spectral radius ``specrad``.
        
        .. todo::
            implementation
        
        """
        raise NotImplementedError()

class orthogonal_reservoir(object):
    """Orthogonal reservoir construction algorithm.
    
    The algorithm starts off with a diagonal matrix, then multiplies
    it with random orthogonal matrices (meaning that the product
    will again be orthogonal). For details, see [TS12]_.
    
    All absolute eigenvalues of the resulting matrix will be
    identical to ``specrad``.
    
    ``density``
        Reservoir density, given in percent (0-100).
    
    """
    def __init__(self, density):
        self._density = density
    
    def __call__(self, out_size, specrad):
        """Return a random, orthogonal reservoir matrix of size
        ``out_size`` and spectral radius ``specrad``."""
        num_entries = int((out_size**2 * self._density)/100.0)
        w = np.random.permutation(np.eye(out_size))
        w = scipy.sparse.csc_matrix(w)
        while w.nnz < num_entries:
            phi = np.random.uniform(0, 2*np.pi)
            i = j = 0
            while i == j:
                i, j = np.random.randint(0, out_size, size=2)
            
            rot = scipy.sparse.eye(out_size, out_size, format='lil')
            rot[i, i] = np.cos(phi)
            rot[j, j] = np.cos(phi)
            rot[i, j] = -np.sin(phi)
            rot[j, i] = np.sin(phi)
            
            if np.random.randint(0, 2) == 0:
                w = rot.tocsc().dot(w)
            else:
                w = w.dot(rot.tocsc())
            
        # Scale to desired spectral radius
        w = specrad * w
        return w

def chain_of_neurons(out_size, specrad):
    """Reservoir setup where the nodes are arranged in a chain.
    Except for the first and last node, each one has exactly one
    predecessor and one successor.
    
    ``out_size``
        Reservoir matrix dimension.
    
    ``specrad``
        Desired spectral radius.
    
    """
    w = scipy.sparse.diags([1.0]*(out_size-1), -1, format='csc')
    w = specrad * w
    return w

def ring_of_neurons(out_size, specrad):
    """Reservoir setup where the nodes are arranged in a ring. Each node
    has exactly one predecessor and one successor.
    
    ``out_size``
        Reservoir matrix dimension.
    
    ``specrad``
        Desired spectral radius.
    
    """
    w = scipy.sparse.diags([1.0]*(out_size-1), -1, format='lil')
    w[0, -1] = 1.0
    w = specrad * w
    return w.tocsc()

## RESERVOIR BASE CLASS ##

class ReservoirNode(object):
    """Reservoir of neurons.
    
    This class is mainly copied from [Oger]_, with some modifications.
    Mainly, the initialization is done differently (yet, compatibility
    maintained through additional keyword arguments) and the reservoir
    state can be computed without affecting the class instance
    (:py:meth:`execute`'s simulate argument).
    
    ``input_dim``
        Number of inputs.
    
    ``output_dim``
        Reservoir size.
    
    ``spectral_radius``
        Spectral radius of the reservoir. The spectral radius of a
        matrix is its largest absolute eigenvalue.
    
    ``nonline_func``
        Reservoir node function, often nonlinear (tanh or sigmoid).
        Default is tanh.
    
    ``reset_states``
        Reset the states to the initial one after a call to
        :py:meth:`execute`. The default is :py:const:`False`, which is
        appropriate in most situations where learning is online.
    
    ``w``
        Reservoir initialization routine. The callback is executed
        on two arguments, the number of nodes and the spectral radius.
        
        The default is :py:func:`sparse_reservoir`, with normally
        distributed weights and density of 20% or ``fan_in_w``
        (see below).
    
    ``w_in``
        Reservoir input weights initialization routine. The callback
        is executed on two arguments, the reservoir and input size.
        
        The default is a dense matrix :py:func:`dense_w_in`, with
        input scaling of 1.0. If ``fan_in_i`` is provided, a sparse
        matrix is created, instead.
    
    ``w_bias``
        Reservoir bias initialization routine. The callback
        is executed on one argument, the reservoir size. The default
        is :py:func:`dense_w_bias`, with the scaling 0.0 (i.e. no bias)
        or ``bias_scaling`` (if provided).
    
    For compatibility with [Oger]_, some additional keyword arguments
    may be provided. These affect the matrix initialization routines
    above and are only used if the respective initialization callback
    is not specified.
    
    ``fan_in_i``
        Density of the input matrix, in percent.
    
    ``fan_in_w``
        Density of the reservoir, in percent.
    
    ``input_scaling``
        Scaling of the input weights.
    
    ``bias_scaling``
        Scaling of the bias.
    
    """
    def __init__(self, input_dim=None, output_dim=None, spectral_radius=0.9,
             nonlin_func=np.tanh, reset_states=False,
             w_bias=None, w=None, w_in=None, **kwargs):
        
        # initialize basic attributes
        self._input_dim = None
        self._output_dim = None
        
        # call set functions for properties
        self.set_input_dim(input_dim)
        self.set_output_dim(output_dim)
        
        # Set all object attributes
        # Spectral radius scaling
        self.spectral_radius = spectral_radius
        # Reservoir states reset
        self.reset_states = reset_states
        # Non-linear function
        if nonlin_func == None:
            self.nonlin_func = np.tanh
        else:
            self.nonlin_func = nonlin_func
        
        # Fields for allocating reservoir weight matrix w, input weight matrix w_in
        # and bias weight matrix w_bias
        self.w_in = np.array([])
        self.w = np.array([])
        self.w_bias = np.array([])
        self.states = np.array([])
        self.initial_state = np.array([])
        # Reservoir initialization
        if w is None:
            if 'fan_in_w' in kwargs:
                warnings.warn('Reservoir init function (w) should be provided')
            density = kwargs.pop('fan_in_w', 20)
            w = sparse_reservoir(density)
        if w_in is None:
            if 'input_scaling' in kwargs or 'fan_in_i' in kwargs:
                warnings.warn('Reservoir input init function (w_in) should be provided')
            input_scaling = kwargs.pop('input_scaling', 1.0)
            if 'fan_in_i' in kwargs:
                density = kwargs.pop('fan_in_i')
                w_in = sparse_w_in(input_scaling, density)
            else:
                w_in = dense_w_in(input_scaling)
        if w_bias is None:
            if 'bias_scaling' in kwargs:
                warnings.warn('Bias init function (w_bias) should be provided')
            
            bias_scaling = kwargs.pop('bias_scaling', 0.0)
            w_bias = dense_w_bias(bias_scaling)
        self.w_initial = w
        self.w_in_initial = w_in
        self.w_bias_initial = w_bias
        
        # Initialization
        self._is_initialized = False
        if input_dim is not None and output_dim is not None:
            # Call the initialize function to create the weight matrices
            self.initialize()
    
    def initialize(self):
        """ Initialize the weight matrices of the reservoir node."""
        if self.input_dim is None:
            raise Exception('Cannot initialize weight matrices: input_dim is not set.')
        
        if self.output_dim is None:
            raise Exception('Cannot initialize weight matrices: output_dim is not set.')
        
        # Initialize input weight matrix
        if callable(self.w_in_initial):
            # If it is a function, call it
            self.w_in = self.w_in_initial(self.output_dim, self.input_dim)
        else:
            # else just copy it
            self.w_in = self.w_in_initial.copy()
        
        # Check if dimensions of the weight matrix match the dimensions of the node inputs and outputs
        if self.w_in.shape != (self.output_dim, self.input_dim):
            exception_str = 'Shape of given w_in does not match input/output dimensions of node. '
            exception_str += 'Input dim: ' + str(self.input_dim) + ', output dim: ' + str(self.output_dim) + '. '
            exception_str += 'Shape of w_in: ' + str(self.w_in.shape)
            raise Exception(exception_str)
        
        # Initialize bias weight matrix
        if callable(self.w_bias_initial):
            # If it is a function, call it
            self.w_bias = self.w_bias_initial(self.output_dim)
        else:
            # else just copy it
            self.w_bias = self.w_bias_initial.copy()
        
        # Check if dimensions of the weight matrix match the dimensions of the node inputs and outputs
        if self.w_bias.shape != (1, self.output_dim):
            exception_str = 'Shape of given w_bias does not match input/output dimensions of node. '
            exception_str += 'Input dim: ' + str(self.input_dim) + ', output dim: ' + str(self.output_dim) + '. '
            exception_str += 'Shape of w_bias: ' + str(self.w_bias.shape)
            raise Exception(exception_str)
        
        # Initialize reservoir weight matrix
        if callable(self.w_initial):
            # If it is a function, call it
            self.w = self.w_initial(self.output_dim, self.spectral_radius)
        else:
            # else just copy it
            self.w = self.w_initial.copy()
        
        # Check if dimensions of the weight matrix match the dimensions of the node inputs and outputs
        if self.w.shape != (self.output_dim, self.output_dim):
            exception_str = 'Shape of given w does not match input/output dimensions of node. '
            exception_str += 'Output dim: ' + str(self.output_dim) + '. '
            exception_str += 'Shape of w: ' + str(self.w.shape)
            raise Exception(exception_str)
        
        self.initial_state = np.zeros((1, self.output_dim))
        self.states = np.zeros((1, self.output_dim))
        
        self._is_initialized = True
    
    def _post_update_hook(self, states, input_, timestep):
        """ Hook which gets executed after the state update equation for every timestep. Do not use this to change the state of the 
            reservoir (e.g. to train internal weights) if you want to use parallellization - use the TrainableReservoirNode in that case.
        """
        pass
    
    def execute(self, x, simulate=False):
        """Executes simulation with input vector ``x``.
        
        ``x``
            Input samples. Variables in columns, observations in rows,
            i.e. if N,M = x.shape then M == ``input_dim``.
        
        ``simulate``
            If :py:const:`True`, the state won't be updated.
        
        """
        # Check if the weight matrices are intialized, otherwise create them
        if not self._is_initialized:
            self.initialize()
        
        # Check input
        self._check_input(x)
        
        # Set the initial state of the reservoir
        # if self.reset_states is true, initialize to zero,
        # otherwise initialize to the last time-step of the previous execute call (for freerun)
        if self.reset_states:
            warnings.warn("Reservoir states are reset - this is quite unusual")
            self.initial_state = np.zeros((1, self.output_dim))
        else:
            self.initial_state = np.atleast_2d(self.states[-1, :])

        steps = x.shape[0]

        # Pre-allocate the state vector, adding the initial state
        states = np.concatenate((self.initial_state, np.zeros((steps, self.output_dim))))

        nonlinear_function_pointer = self.nonlin_func

        # Loop over the input data and compute the reservoir states
        for i in range(steps):
            states[i + 1, :] = nonlinear_function_pointer(self.w*states[i, :] + self.w_in* x[i, :] + self.w_bias)
            self._post_update_hook(states, x, i)

        # Strip the initial state
        states = states[1:, :]
        # Save the state for re-initialization unless in simulatino
        if not simulate:
            self.states = states
        # Return the updated reservoir state
        return states
    
    def reset(self):
        """Reset the reservoir states to the initial value."""
        self.states = np.zeros((1, self.output_dim))
    
    def __call__(self, x, *args, **kwargs):
        """Calling an instance of `Node` is equivalent to calling
        its `execute` method."""
        return self.execute(x, *args, **kwargs)
    
    ## GETTERS, SETTERS, LEFTOVERS FROM MDP/OGER ##
    
    def get_input_dim(self):
        """Return input dimensions."""
        return self._input_dim
    
    def set_input_dim(self, dim):
        """Set input dimensions.
        
        Perform sanity checks and then calls ``self._set_input_dim(dim)``, which
        is responsible for setting the internal attribute ``self._input_dim``.
        Note that subclasses should overwrite `self._set_input_dim`
        when needed.
        """
        if dim is None:
            pass
        elif (self._input_dim is not None) and (self._input_dim != dim):
            msg = ("Input dim are set already (%d) "
                   "(%d given)!" % (self.input_dim, dim))
            raise Exception(msg)
        else:
            self._input_dim = dim
    
    input_dim = property(get_input_dim,
                         set_input_dim,
                         doc="Input dimensions")
    
    def get_output_dim(self):
        """Return output dimensions."""
        return self._output_dim
    
    def set_output_dim(self, dim):
        """Set output dimensions.

        Perform sanity checks and then calls ``self._set_output_dim(dim)``, which
        is responsible for setting the internal attribute ``self._output_dim``.
        Note that subclasses should overwrite `self._set_output_dim`
        when needed.
        """
        self._output_dim = dim
    
    output_dim = property(get_output_dim,
                          set_output_dim,
                          doc="Output dimensions")
    
    ### check functions
    def _check_input(self, x):
        """Check if input data ``x`` matches the reservoir requirements."""
        # check input rank
        if not x.ndim == 2:
            error_str = "x has rank %d, should be 2" % (x.ndim)
            raise Exception(error_str)
        
        # set the input dimension if necessary
        if self.input_dim is None:
            self.input_dim = x.shape[1]
        
        # check the input dimension
        if not x.shape[1] == self.input_dim:
            error_str = "x has dimension %d, should be %d" % (x.shape[1],
                                                              self.input_dim)
            raise Exception(error_str)
        
        if x.shape[0] == 0:
            error_str = "x must have at least one observation (zero given)"
            raise Exception(error_str)
    
    def __str__(self):
        return str(type(self).__name__)
    
    def __repr__(self):
        # print input_dim, output_dim
        name = type(self).__name__
        inp = "input_dim=%s" % str(self.input_dim)
        out = "output_dim=%s" % str(self.output_dim)
        args = ', '.join((inp, out))
        return name + '(' + args + ')'
    
    def copy(self):
        """Return a deep copy of the node."""
        return _copy.deepcopy(self)
    
    def save(self, filename, protocol=-1):
        """Save a pickled serialization of the node to `filename`.
        If `filename` is None, return a string.

        Note: the pickled `Node` is not guaranteed to be forwards or
        backwards compatible."""
        if filename is None:
            return _cPickle.dumps(self, protocol)
        else:
            # if protocol != 0 open the file in binary mode
            mode = 'wb' if protocol != 0 else 'w'
            with open(filename, mode) as flh:
                _cPickle.dump(self, flh, protocol)

class SparseReservoirNode(ReservoirNode):
    """
    
    .. deprecated:: 1.0
        Use :py:class:`ReservoirNode` instead
    
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("This class is deprecated. Use 'SparseReservoirNode' instead")
        super(SparseReservoirNode, self).__init__(*args, **kwargs)

## RLS ##

#class PlainRLS: # For old pickled instances, the class must not be new-style
class PlainRLS(object):
    """Compute online least-square, multivariate linear regression.
    
    The interface is based on [MDP]_ and [Oger]_ with two major
    differences:
    
    1) There's no inheritance from any [MDP]_ or [Oger]_ class
    2) The error as well as a target may be passed to the :py:meth:`train`
       function
    
    The implementation follows the algorithm given in [FB98]_
    (least-squares filter).
    
    ``with_bias``
        Add a constant to the linear equation. This internally increases
        the input dimension by one, whereas the added input is always
        one. Generally, the target function is
        :math:`f(x) = \sum_{i=0}^N a_i x_i`. With `with_bias=True` it
        will be :math:`f(x) = \sum_{i=0}^N a_i x_i + a_{N+1}` instead.
        (default :py:const:`True`)
    
    ``input_dim``
        Dimension of the input (number of observations per sample)
    
    ``output_dim``
        Dimension of the output (Regression order)
    
    ``lambda_``
        Forgetting factor. Controlls how much the training focuses
        on recent samples (see [FB98]_ for details).
        (default 1.0, corresponds to offline learning)
        
        "Roughly speaking :math:`\\frac{1}{1-\lambda}` is a measure of the
        *memory* of the algorithm. The case of :math:`\lambda = 1`
        corresponds to *infinite memory*." [FB98]_
        
    
    """
    def __init__(self, input_dim, output_dim, with_bias=True, lambda_=1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_ = lambda_
        self.with_bias = with_bias
        if self.with_bias:
            input_dim += 1
        self.beta = np.zeros((input_dim, self.output_dim))
        self._psi_inv = np.eye(input_dim, input_dim) * 10000.0
        self._stop_training = False

    def train(self, sample, trg=None, err=None, d=None, e=None):
        """Train the regression on one or more samples.
        
        ``sample``
            Input samples. Array of size (K, input_dim)
        
        ``train``
            Sample target. Array of size (K, output_dim)
        
        ``err``
            Sample error terms. Array of size (K, output_dim)
        
        """
        if self._stop_training:
            return
        
        if d is not None:
            warnings.warn("Use of argument 'd' is deprecated. Use 'trg' instead.")
            trg = d
        
        if e is not None:
            warnings.warn("Use of argument 'e' is deprecated. Use 'err' instead.")
            err = e
        
        if self.with_bias:
            sample = self._add_constant(sample)
        for i in range(sample.shape[0]):
            # preliminaries
            sample_i = np.atleast_2d(sample[i]).T
            psi_x = self._psi_inv.dot(sample_i)
            gain = psi_x / (self.lambda_ + sample_i.T.dot(psi_x))
            # error
            if err is None:
                trg_i = np.atleast_2d(trg[i]).T
                pred = self.beta.T.dot(sample_i)
                err_i = trg_i - pred
            else:
                err_i = np.atleast_2d(err[i]).T
            # update
            self.beta += gain.dot(err_i.T)
            self._psi_inv -= gain.dot(sample_i.T.dot(self._psi_inv))
            self._psi_inv /= self.lambda_
    
    def __call__(self, x):
        """Evaluate the linear approximation on some point ``x``.
        """
        if self.with_bias:
            x = self._add_constant(x)
        return self.beta.T.dot(x.T).T
    
    def _add_constant(self, x):
        """Add a constant term to the vector 'x'.
        x -> [1 x]
        """
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    
    def save(self, pth):
        """Save the regression state in a file.
        
        To load the state, use pickle, as show below.
        
        >>> import numpy as np
        >>> import tempfile, os, pickle
        >>> fh,pth = tempfile.mkstemp()
        >>> r1 = PlainRLS(input_dim=5, output_dim=2, lambda_=0.9, with_bias=True)
        >>> r1.train(np.random.normal(size=(100,5)), e=np.random.normal(size=(100,2)))
        >>> r1.save(pth)
        >>> f = open(pth,'r')
        >>> r2 = pickle.load(f)
        >>> f.close()
        >>> os.unlink(pth)
        >>> (r2.beta == r1.beta).all()
        True
        >>> (r2._psi_inv == r1._psi_inv).all()
        True
        >>> r2.input_dim == r1.input_dim
        True
        >>> r2.output_dim == r1.output_dim
        True
        >>> r2.lambda_ == r1.lambda_
        True
        >>> r2.with_bias == r1.with_bias
        True
        
        """
        f = open(pth, 'w')
        _cPickle.dump(self, f)
        f.close()
    
    def __repr__(self):
        return 'PlainRLS(with_bias=%r, input_dim=%i, output_dim=%i, lambda_=%f)' % (self.with_bias, self.input_dim, self.output_dim, self.lambda_)
    
    def stop_training(self):
        """Disable filter adaption for future calls to ``train``."""
        self._stop_training = True
    
    def copy(self):
        """Return a deep copy of the node."""
        return _copy.deepcopy(self)
    

class StabilizedRLS(PlainRLS):
    """Compute online least-square, multivariate linear regression.
    
    Identical to :py:class:`PlainRLS`, except that the internal matrices
    are computed on the lower triangular part. This ensures symmetrical
    matrices even with floating point operations. If unsure, use this
    implementation instead of :py:class:`PlainRLS`.
    
    """
    def train(self, sample, trg=None, err=None, d=None, e=None):
        """Train the regression on one or more samples.
        
        ``sample``
            Input samples. Array of size (K, input_dim)
        
        ``trg``
            Sample target. Array of size (K, output_dim)
        
        ``err``
            Sample error terms. Array of size (K, output_dim)
        
        """
        if self._stop_training:
            return
        
        if d is not None:
            warnings.warn("Use of argument 'd' is deprecated. Use 'trg' instead.")
            trg = d
        
        if e is not None:
            warnings.warn("Use of argument 'e' is deprecated. Use 'err' instead.")
            err = e
        
        if self.with_bias:
            sample = self._add_constant(sample)
        
        for i in range(sample.shape[0]):
            # preliminaries
            sample_i = np.atleast_2d(sample[i]).T
            psi_x = self._psi_inv.dot(sample_i)
            gain = 1.0 / (self.lambda_ + sample_i.T.dot(psi_x)) * psi_x
            # error
            if err is None:
                trg_i = np.atleast_2d(trg[i]).T
                pred = self.beta.T.dot(sample_i)
                err_i = trg_i - pred
            else:
                err_i = np.atleast_2d(err[i]).T
            # update
            self.beta += gain.dot(err_i.T)
            tri = np.tril(self._psi_inv)
            tri -= np.tril(gain*psi_x.T)
            tri /= self.lambda_
            #self._psi_inv = tri + tri.T - np.diag(tri.diagonal())
            # FIXME: (numpy bug) tri.diagonal() introduces a memory leak
            self._psi_inv = np.tril(tri, -1).T + tri
    
    def __repr__(self):
        return 'StabilizedRLS(with_bias=%r, input_dim=%i, output_dim=%i, lambda_=%f)' % (self.with_bias, self.input_dim, self.output_dim, self.lambda_)


## RESERVOIR MEMORY MEASUREMENT ##

def reservoir_memory(reservoir, max_settling_time=10000):
    """Measure the memory capacity of a ``reservoir``. Make sure, the
    reservoir is initialized. The settling time is measured, which is
    the number of steps it takes until the reservoir has converged after
    some non-zero input.
    """
    res_eval = reservoir.copy()
    res_eval.reset_states = True
    step_input = np.zeros((max_settling_time, reservoir.get_input_dim()))
    step_input[0] += 1.0
    ret = res_eval(step_input)
    hist = (abs(ret[1:, :] - ret[:-1, :]) < 1e-5).sum(axis=1)
    settling_time = hist.argmax()
    return settling_time

def query_reservoir_memory(reservoir, steps=1000, max_settling_time=10000):
    """Compute the impulse response settling time for several spectral
    radii. A function is returned for querying the spectral radius given
    a minimal settling time.
    
    .. deprecated:: 1.0
        The same use-case is appraoched by :py:func`find_radius_for_mc`
        and :py:func:`reservoir_memory`, but finer grained.
    
    """
    res_eval = reservoir.copy()
    res_eval.reset_states = True
    rad = map(lambda i: 1.0*i/steps, range(1, steps+1))
    rad = map(float, rad)
    step_input = np.zeros((max_settling_time, res_eval.get_input_dim()))
    step_input[0] += 1.0
    data = []
    for rad0, rad1 in zip([res_eval.spectral_radius] + rad, rad):
        res_eval.w *= rad1/rad0
        res_eval.spectral_radius = rad1
        ret = res_eval(step_input)
        hist = (abs(ret[1:, :] - ret[:-1, :]) < 1e-5).sum(axis=1)
        settling_time = hist.argmax()
        data.append((rad1, settling_time))
    
    data.sort(key=lambda (i, j): j)
    
    def query(min_steps):
        """Get the spectral radius and number of steps until the impulse
        response has converged for a minimal number of such steps.
        """
        for rad, steps in data:
            if steps >= min_steps:
                return rad, steps
    
    return query

def find_radius_for_mc(reservoir, num_steps, tol=1.0, max_settling_time=10000, tol_settling=0.5, num_iter=100):
    """Find a spectral radius for ``reservoir`` such that the impulse
    response time is ``num_steps`` with tolerance ``tol``.
    
    This implementation linearizes the memory capacity function locally
    and uses binary search to approach the target value. If you look
    for a single value (not the whole characteristics), this function
    is usually faster than :py:func:`query_reservoir_memory`.
    
    ``max_settling_time`` sets the maximum time after which the
    reservoir should have settled.
    
    ``tol_settling`` and ``num_iter`` are search abortion criteria which
    break, if the settling time doesn't change too much or too many
    iterations have been run. If any of the criteria is met, an
    Exception is thrown.
    
    """
    assert max_settling_time > num_steps + tol
    assert tol_settling < tol
    res_eval = reservoir.copy()
    res_eval.reset_states = True
    step_input = np.zeros((max_settling_time, res_eval.get_input_dim()))
    step_input[0] += 1.0
    
    # initial loop values
    rad_lo, rad_hi, rad = 0.0, 1.0, 0.5
    
    # loop
    settling_time = float('inf')
    while True:
        
        # set up new reservoir
        res_eval.w *= rad / res_eval.spectral_radius
        res_eval.spectral_radius = rad
        
        # evaluate MC
        ret = res_eval(step_input)
        hist = (abs(ret[1:, :] - ret[:-1, :]) < 1e-5).sum(axis=1)
        prev_settling_time = settling_time
        settling_time = hist.argmax()
        
        if settling_time < num_steps:
            rad_lo = rad
            rad = rad + (rad_hi - rad)/2.0
        else:
            rad_hi = rad
            rad = rad - (rad - rad_lo)/2.0
        
        if abs(num_steps - settling_time) > tol:
            break
            
        if num_iter < 0 or abs(settling_time - prev_settling_time) < tol_settling:
            raise Exception('No solution found')
        
        # continue
        num_iter -= 1
    
    return rad



class ESN:
    """
    
    .. todo::
        (Experimental) Interface, Implementation, Documentation
    
    """
    def __init__(self, reservoir, readout):
        pass
    def execute(self):
        pass
    def train(self):
        pass
    def _preprocessing(self):
        pass
    def _postprocessing(self):
        pass

class ActionReadoutNode(object):
    """A readout node from the reservoir"""
    def __init__(self, input_dim, output_dim, w=None, w_scaling=1.0, with_bias=True, alpha=1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w_scaling = w_scaling
#        if not callable(alpha):
#            alpha = self.const(alpha)
        self.alpha = alpha # learning rate
        self.with_bias = with_bias
        if self.with_bias:
            input_dim += 1
        if w is None:
            w = np.random.uniform(size=(self.input_dim+int(with_bias), self.output_dim))*2.0-1.0 * self.w_scaling
        if callable(w):
            # If it is a function, call it
            self.w = w(self.input_dim+int(with_bias), self.output_dim)
        else:
            # else just copy it
            self.w = w.copy()
        self._stop_training = False
    
#    @staticmethod
#    def const(a):
#        def func(t): return a
#        return func
    
    def train(self, sample, deriv, t):
        """Train the regression on one or more samples.
        
        ``sample``
            Input samples. A concatenation of reservoir-states, sensor-states and actions
        ``deriv``
            Derivative of action according to HDPy.ADHDP
        
        """
        if self._stop_training:
            return
                
        if self.with_bias:
            sample = self._add_constant(sample)
        
        # update:
        self.w += self.alpha * np.outer(sample, deriv)
    
    def __call__(self, x):
        """Evaluate the linear approximation on some point ``x``.
        """
        if self.with_bias:
            x = self._add_constant(x)
        return self.w.T.dot(x.T).T
    
    def _add_constant(self, x):
        """Add a constant term to the vector 'x'.
        x -> [1 x]
        """
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    
    def save(self, pth):
        """Save the node in a file.
        """
        f = open(pth, 'w')
        _cPickle.dump(self, f)
        f.close()
    
    def __repr__(self):
        return 'ActionReadoutNode(with_bias=%r, input_dim=%i, output_dim=%i, w_scaling=%f, alpha=%f)' % (self.with_bias, self.input_dim, self.output_dim, self.w_scaling, self.alpha)
    
    def stop_training(self):
        """Disable filter adaption for future calls to ``train``."""
        self._stop_training = True
    
    def copy(self):
        """Return a deep copy of the node."""
        return _copy.deepcopy(self)


class RewardModulatedHebbianLearningNode(object):
    """Hebbian learning node with reward modulation.
    ``input_dim``
        Number of inputs.
    ``output_dim``
        Number of outputs.
    ``learning_rate``
        Step size of the weight updates.
    ``w``
        Input weights initialization routine. The callback
        is executed on two arguments, the output and input size.
        The default is a sparse matrix :py:func:`sparse_w_in`, with
        input scaling of 1.0 and density 100. If ``fan_in_w`` is provided,
        a sparse matrix is created, instead.
    ``density``
        Density of the weights, in percent.
    ``input_scaling``
        Scaling of the input weights.
    """
    def __init__(self, input_dim=None, output_dim=None, learning_rate=0.1, w=None, with_bias=True, **kwargs):
        
        # initialize basic attributes
        self._input_dim = None
        self._output_dim = None
        self.with_bias = with_bias
        
        # call set functions for properties
        self.set_input_dim(input_dim)
        self.set_output_dim(output_dim)
        
        # Set all object attributes
        # Learning rate
        self.learning_rate = learning_rate
        
        # Fields for allocating weight matrix w
        self.w = np.array([])
        # Reservoir initialization
        if w is None:
            if 'density' in kwargs:
                warnings.warn('Reservoir init function (w) should be provided')
            density = kwargs.pop('density', 100)
            input_scaling = kwargs.pop('w_scaling', 1.0)
            w = sparse_w_in(input_scaling, density)
        self.w_initial = w
        
        # Initialization
        self._is_initialized = False
        if input_dim is not None and output_dim is not None:
            # Call the initialize function to create the weight matrices
            self.initialize()

        self._stop_training = False
    
    def initialize(self):
        """ Initialize the weight matrices of the reservoir node."""
        if self.input_dim is None:
            raise Exception('Cannot initialize weight matrices: input_dim is not set.')
        
        if self.output_dim is None:
            raise Exception('Cannot initialize weight matrices: output_dim is not set.')
        
        # Initialize weight matrix
        if callable(self.w_initial):
            # If it is a function, call it
            self.w = self.w_initial(self.output_dim, self.input_dim+int(self.with_bias))
        else:
            # else just copy it
            self.w = self.w_initial.copy()
        
        # Check if dimensions of the weight matrix match the dimensions of the node inputs and outputs
        if self.w.shape != (self.output_dim, self.input_dim+int(self.with_bias)):
            exception_str = 'Shape of given w does not match input/output dimensions of node. '
            exception_str += 'Input dim: ' + str(self.input_dim+int(self.with_bias)) + ', output dim: ' + str(self.output_dim) + '. '
            exception_str += 'Shape of w: ' + str(self.w.shape)
            raise Exception(exception_str)
        
        self._is_initialized = True
    
    def execute(self, x):
        """Executes simulation with input vector ``x``.
        ``x``
            Input samples. Variables in columns, observations in rows,
            i.e. if N,M = x.shape then M == ``input_dim``.
        """
        # Check if the weight matrices are initialized, otherwise create them
        if not self._is_initialized:
            self.initialize()
        
        # Check input
        self._check_input(x)
        
        # Add bias
        if self.with_bias:
            x = self._add_constant(x)

        # Apply weights to input
        output = self.w * x.T

        # Return the output
        return output.T
    
    def __call__(self, x, *args, **kwargs):
        """Calling an instance of `Node` is equivalent to calling
        its `execute` method."""
        return self.execute(x, *args, **kwargs)

    def train(self, x, y, reward):
        """Train the network on one or more samples.
        ``x``
            Input samples. Array of size (K, input_dim)
        ``y``
            Output samples. Array of size (K, output_dim)
        ``reward``
            Reward of the samples. Array of size (K,)
        """
        if self._stop_training:
            return
        
        # Check input
        self._check_input(x)
        
        # Add bias
        if self.with_bias:
            x = self._add_constant(x)
        
        # Update the matrix
        for i in range(x.shape[0]):
            self.w = scipy.sparse.csc_matrix((1.0-self.learning_rate) * self.w + self.learning_rate * reward[i] * np.outer(y[i,:], x[i,:]))
            self.w = (self.w / self.w.sum()) * self.w_initial._scaling
    
    ## GETTERS, SETTERS, LEFTOVERS FROM MDP/OGER ##
    
    def get_input_dim(self):
        """Return input dimensions."""
        return self._input_dim
    
    def set_input_dim(self, dim):
        """Set input dimensions.
        Perform sanity checks and then calls ``self._set_input_dim(dim)``, which
        is responsible for setting the internal attribute ``self._input_dim``.
        Note that subclasses should overwrite `self._set_input_dim`
        when needed.
        """
        if dim is None:
            pass
        elif (self._input_dim is not None) and (self._input_dim != dim):
            msg = ("Input dim are set already (%d) "
                   "(%d given)!" % (self.input_dim, dim))
            raise Exception(msg)
        else:
            self._input_dim = dim
    
    input_dim = property(get_input_dim,
                         set_input_dim,
                         doc="Input dimensions")
    
    def get_output_dim(self):
        """Return output dimensions."""
        return self._output_dim
    
    def set_output_dim(self, dim):
        """Set output dimensions.
        Perform sanity checks and then calls ``self._set_output_dim(dim)``, which
        is responsible for setting the internal attribute ``self._output_dim``.
        Note that subclasses should overwrite `self._set_output_dim`
        when needed.
        """
        self._output_dim = dim
    
    output_dim = property(get_output_dim,
                          set_output_dim,
                          doc="Output dimensions")
    
    ### check functions
    def _check_input(self, x):
        """Check if input data ``x`` matches the requirements."""
        # check input rank
        if not x.ndim == 2:
            error_str = "x has rank %d, should be 2" % (x.ndim)
            raise Exception(error_str)
        
        # set the input dimension if necessary
        if self.input_dim is None:
            self.input_dim = x.shape[1]
        
        # check the input dimension
        if not x.shape[1] == self.input_dim:
            error_str = "x has dimension %d, should be %d" % (x.shape[1],
                                                              self.input_dim)
            raise Exception(error_str)
        
        if x.shape[0] == 0:
            error_str = "x must have at least one observation (zero given)"
            raise Exception(error_str)  

    def _add_constant(self, x):
        """Add a constant term to the vector 'x'.
        x -> [1 x]
        """
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    
    def __str__(self):
        return str(type(self).__name__)
    
    def __repr__(self):
        # print input_dim, output_dim
        name = type(self).__name__
        inp = "input_dim=%s" % str(self.input_dim)
        out = "output_dim=%s" % str(self.output_dim)
        args = ', '.join((inp, out))
        return name + '(' + args + ')'

    def stop_training(self):
        """Disable filter adaption for future calls to ``train``."""
        self._stop_training = True
    
    def copy(self):
        """Return a deep copy of the node."""
        return _copy.deepcopy(self)
    
    def save(self, filename, protocol=-1):
        """Save a pickled serialization of the node to `filename`.
        If `filename` is None, return a string.
        Note: the pickled `Node` is not guaranteed to be forwards or
        backwards compatible."""
        if filename is None:
            return _cPickle.dumps(self, protocol)
        else:
            # if protocol != 0 open the file in binary mode
            mode = 'wb' if protocol != 0 else 'w'
            with open(filename, mode) as flh:
                _cPickle.dump(self, flh, protocol)


