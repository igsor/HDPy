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


"""
import warnings
import numpy as np
import scipy.sparse
import copy as _copy
import cPickle as _cPickle

## RESERVOIR BASE CLASS ##

class ReservoirNode(object):
    """
    
    .. todo::
        Documentation:
        
        * Density of input and reservoir matrix
        * Execution is able to simulate a step
        * Matrices outsourced to functions; mention defaults
    
    """
    def __init__(self, input_dim=None, output_dim=None, spectral_radius=0.9,
             nonlin_func=np.tanh, reset_states=True,
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
            warnings.warn('Reservoir init function (w) should be provided')
            density = kwargs.pop('fan_in_w', 20)
            w = lambda N, srad: sparse_reservoir(N, srad, density, rnd_fu=np.random.normal)
        if w_in is None:
            warnings.warn('Reservoir input init function (w_in) should be provided')
            input_scaling = kwargs.pop('input_scaling', 1.0)
            if 'fan_in_i' in kwargs:
                density = kwargs.pop('fan_in_i')
                w_in = lambda N, M: sparse_w_in(N, M, input_scaling, density)
            else:
                w_in = lambda N, M: dense_w_in(N, M, input_scaling)
        if w_bias is None:
            warnings.warn('Bias init function (w_bias) should be provided')
            bias_scaling = kwargs.pop('bias_scaling', 0.0)
            w_bias = lambda N: dense_w_bias(N, bias_scaling)
        
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
    
    def _execute(self, x, simulate=False):
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


## RESERVOIR SETUP FUNCTIONS ##

def dense_w_in(out_size, in_size, scaling, rnd_fu=lambda **kwargs: np.random.uniform(**kwargs)*2.0-1.0):
    """Dense input matrix.
    
    .. todo::
        documentation
    
    ``out_size``
        
    
    ``in_size``
        
    
    ``scaling``
        
    
    ``rnd_fu``
        
    
    """
    return scaling * rnd_fu(size=(out_size, in_size))

def sparse_w_in(out_size, in_size, scaling, density, rnd_fu=lambda **kwargs: np.random.uniform(**kwargs)*2.0-1.0):
    """Sparse, random reservoir input. The density is exactly as
    defined by ``density``, given in percent (0-100). The values are
    drawn from an uniform distribution.
    
    ``out_size``
        Number of reservoir nodes.
    
    ``in_size``
        Number of input nodes.
    
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
    # Initialize reservoir weight matrix
    nrentries = np.int32(out_size * in_size * density / 100.0)
    idx = np.random.choice(out_size*in_size, nrentries, False)
    locs = np.array(zip(*map(lambda i: (i/in_size, i%in_size), idx)))
    datavec = rnd_fu(size=nrentries)
    w = scaling * scipy.sparse.csc_matrix((datavec, locs), shape=(out_size, in_size))
    return w

def dense_w_bias(out_size, scaling, rnd_fu=lambda **kwargs: np.random.rand(**kwargs)*2.0-1.0):
    """Dense reservoir node bias. The weights are picked from a random
    number generator ``rnd_fu`` and scaled according to ``scaling``.
    
    ``out_size``
        Number of reservoir nodes.
    
    ``scaling``
        Scale of the bias weights
    
    ``rnd_fu``
        Random number generator for the weights. Default is normal
        distribution in [-1.0, 1.0].
    """
    return scaling * rnd_fu(size=(1, out_size))

def sparse_reservoir(out_size, specrad, density, rnd_fu=np.random.normal):
    """:py:meth:`Oger.nodes.SparseReservoirNode.sparse_w` with
    density correction. Now, the density is exactly as set by
    ``fan_in_w``, given in percent (0-100). The non-zero locations
    are uniformly distributed and the values are drawn from a normal
    distribution by default.
    
    ``out_size``
        Number of reservoir nodes.
    
    ``specrad``
        Spectral radius.
    
    ``rnd_fu``
        Random number generator for the weight values. Must take
        the scalar argument *size* that determines the length of
        the generated random number list. The default is the normal
        distribution.
    
    """
    from scipy.sparse.linalg import eigs, ArpackNoConvergence
    converged = False
    # Initialize reservoir weight matrix
    nrentries = int((out_size**2 * density)/100.0)
    out_size = int(out_size)
    # Keep generating random matrices until convergence
    num_iter = 1000
    while not converged:
        try:
            idx = np.random.choice(out_size**2, nrentries, False)
            locs = np.array(zip(*map(lambda i: (i/out_size, i%out_size), idx)))
            datavec = rnd_fu(size=nrentries)
            w = scipy.sparse.csc_matrix((datavec, locs), shape=(out_size, out_size))
            eigvals = eigs(w, return_eigenvectors=False, k=3)
            converged = True
            w *= (specrad / np.amax(np.absolute(eigvals)))
        except ArpackNoConvergence:
            num_iter -= 1
            if num_iter < 1:
                raise Exception('Reservoir matrix could not be built')
    return w

def orthogonal_reservoir(out_size, specrad, density, rnd_fu=None):
    """Orthogonal reservoir construction algorithm.
    
    The algorithm starts off with a diagonal matrix, then multiplies
    it with random orthogonal matrices (meaning that the product
    will again be orthogonal). For details, see [TS12]_.
    
    All absolute eigenvalues of the resulting matrix will be
    identical to ``specrad``.
    
    ``out_size``
        Reservoir matrix dimension.
    
    ``specrad``
        Desired spectral radius.
    
    ``rnd_fu``
        This argument is obsolete in this function, but present for
        compatibility only.
    
    """
    num_entries = int((out_size**2 * density)/100.0)
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

def chain_of_neurons(out_size, specrad, rnd_fu=None):
    """
    
    ``out_size``
        Reservoir matrix dimension.
    
    ``specrad``
        Desired spectral radius.
    
    ``rnd_fu``
        This argument is obsolete in this function, but present for
        compatibility only.
    
    """
    w = scipy.sparse.diags([1.0]*(out_size-1), -1, format='csc')
    w = specrad * w
    return w

def ring_of_neurons(out_size, specrad, rnd_fu=None):
    """
    
    ``out_size``
        Reservoir matrix dimension.
    
    ``specrad``
        Desired spectral radius.
    
    ``rnd_fu``
        This argument is obsolete in this function, but present for
        compatibility only.
    
    """
    w = scipy.sparse.diags([1.0]*(out_size-1), -1, format='lil')
    w[0, -1] = 1.0
    w = specrad * w
    return w.tocsc()


## RLS ##

#class PlainRLS: # For old pickled instances, the class must not be new-style
class PlainRLS(object):
    """Compute online least-square, multivariate linear regression.
    
    The interface is based on :py:class:`OnlineLinearRegression` with
    two major changes:
    
    1) There's no inheritance from any mdp or Oger class
    2) The error as well as a target may be passed to the :py:meth:`train`
       function
    
    The implementation follows the original algorithm given in [FB98]_.
    
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

class StabilizedRLS(PlainRLS):
    """Compute online least-square, multivariate linear regression.
    
    Identical to :py:class:`PlainRLS`, except that the internal matrices
    are computed on the lower triangular part. This ensures symmetrical
    matrices even with floating point operations. If unsure, use this
    implementation instead of :py:class:`PlainRLS`.
    
    """
    def train(self, sample, trg=None, err=None):
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
    """Find a spectral radius for ``reservoir`` such that the step
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
