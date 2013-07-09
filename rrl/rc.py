"""
Reservoir Computing code.

These mostly are some bugfixes and extensions on the Oger/mdp framework.
At some point, some of this code might be merged back to Oger.

"""

import Oger
import mdp
import warnings
import numpy as np

## RESERVOIR BASE CLASS ##

class SparseReservoirNode(Oger.nodes.ReservoirNode):
    """:py:class:`Oger.nodes.SparseReservoirNode` with some corrections
    and extensions:
    
    * Density of input and reservoir matrix
    * Execution is able to simulate a step
    
    """
    def __init__(self, input_dim=None, output_dim=None, spectral_radius=0.9,
             nonlin_func=np.tanh, reset_states=True, bias_scaling=0, input_scaling=1, dtype='float64', _instance=0,
             w_bias=None, fan_in_w=20, fan_in_i=100, w=None, w_in=None):
        
        if w is None: w = self.sparse_w
        if w_in is None: w_in = self.sparse_w_in
        self.fan_in_w = fan_in_w
        self.fan_in_i = fan_in_i
        
        super(SparseReservoirNode, self).__init__(input_dim=input_dim, output_dim=output_dim, spectral_radius=spectral_radius,
                 nonlin_func=nonlin_func, reset_states=reset_states, bias_scaling=bias_scaling, input_scaling=input_scaling, dtype=dtype, _instance=_instance,
                 w_in=w_in, w_bias=w_bias, w=w)

    def sparse_w(self, out_size, specrad, rnd_fu=np.random.normal):
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
        import scipy.sparse
        from scipy.sparse.linalg import eigs
        converged = False
        # Initialize reservoir weight matrix
        nrentries = int((out_size**2 * self.fan_in_w)/100.0)
        out_size = int(out_size)
        # Keep generating random matrices until convergence
        numIter=1000
        while not converged and numIter > 0:
            numIter -= 1
            try:
                idx = np.random.choice(out_size**2, nrentries, False)
                ij = np.array(zip(*map(lambda i: (i/out_size, i%out_size), idx)))
                datavec = rnd_fu(size=nrentries)
                w = scipy.sparse.csc_matrix((datavec, ij), shape=(out_size, out_size))
                we = eigs(w,return_eigenvectors=False,k=3)
                converged = True
                w *= (specrad / np.amax(np.absolute(we)))
            except:
                pass
        if numIter <= 0: raise Exception('Reservoir matrix could not be built')
        return w
    
    def dense_w_in(self, out_size, in_size, scaling, rnd_fu=lambda size: np.random.uniform(size=size)*2.0-1.0):
        return scaling * rnd_fu(size=(self.output_dim, self.input_dim))
    
    def sparse_w_in(self, out_size, in_size, scaling, rnd_fu=lambda size: np.random.uniform(size=size)*2.0-1.0):
        """:py:meth:`Oger.nodes.SparseReservoirNode.sparse_w_in` with
        density correction. Now, the density is exactly as defined by
        ``fan_in_i``, given in percent (0-100). The values are drawn
        from an uniform distribution.
        
        ``out_size``
            Number of reservoir nodes.
        
        ``in_size``
            Number of input nodes.
        
        ``scaling``
            Scaling of input weights.
        
        ``rnd_fu``
            Random number generator for the weight values. Must take
            the scalar argument *size* that determines the length of
            the generated random number list. The default is the uniform
            distribution between -1.0 and 1.0.
        
        """
        import scipy.sparse
        # Initialize reservoir weight matrix
        nrentries = np.int32(out_size * in_size * self.fan_in_i / 100.0)
        idx = np.random.choice(out_size*in_size, nrentries, False)
        ij = np.array(zip(*map(lambda i: (i/in_size, i%in_size), idx)))
        datavec = rnd_fu(size=nrentries)
        w = scaling * scipy.sparse.csc_matrix((datavec, ij),dtype=self._dtype, shape=(out_size, in_size))
        return w
    
    def reset(self):
        """Reset the reservoir states to the initial value."""
        self.states = mdp.numx.zeros((1, self.output_dim))
    
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

        # Set the initial state of the reservoir
        # if self.reset_states is true, initialize to zero,
        # otherwise initialize to the last time-step of the previous execute call (for freerun)
        if self.reset_states:
            self.initial_state = np.zeros((1, self.output_dim))
        else:
            self.initial_state = np.atleast_2d(self.states[-1, :])

        steps = x.shape[0]

        # Pre-allocate the state vector, adding the initial state
        states = np.concatenate((self.initial_state, np.zeros((steps, self.output_dim))))

        nonlinear_function_pointer = self.nonlin_func

        # Loop over the input data and compute the reservoir states
        for n in range(steps):
            states[n + 1, :] = nonlinear_function_pointer(self.w*states[n, :] + self.w_in* x[n, :] + self.w_bias)
            self._post_update_hook(states, x, n)

        # Strip the initial state
        states = states[1:,:]
        # Save the state for re-initialization unless in simulatino
        if not simulate: self.states = states
        # Return the updated reservoir state
        return states

## SPECIFIC RESERVOIR SETUP ##

def OrthogonalReservoir(self, out_size, specrad, rnd_fu=None):
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
    import scipy.sparse
    from math import cos,sin,pi
    num_entries = int((out_size**2 * self.fan_in_w)/100.0)
    W = np.random.permutation(np.eye(out_size))
    W = scipy.sparse.csc_matrix(W)
    while W.nnz < num_entries:
        phi = np.random.uniform(0, 2*pi)
        h=k=0
        while h==k:
            h,k = np.random.randint(0, out_size, size=2)
        
        Q = scipy.sparse.eye(out_size, out_size, format='lil')
        Q[h,h] = cos(phi)
        Q[k,k] = cos(phi)
        Q[h,k] = -sin(phi)
        Q[k,h] = sin(phi)
        
        if np.random.randint(0,2) == 0:
            W = Q.tocsc().dot(W)
        else:
            W = W.dot(Q.tocsc())
    
    # Scale to desired spectral radius
    W = specrad * W
    return W

def ChainOfNeurons(self, out_size, specrad, rnd_fu=None):
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

def RingOfNeurons(self, out_size, specrad, rnd_fu=None):
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
    w[0,-1] = 1.0
    w = specrad * w
    return w.tocsc()

## FLOW ##

class FreerunFlow(Oger.nodes.FreerunFlow):
    """:py:class:`Oger.nodes.FreerunFlow` with corrections and
    extensions.
    
    .. todo::
        The class should be usable within the cross-validation framework
    
    .. note::
        This class is a stub, there's no implementation. However,
        changes are likely to happen, so when working with this library,
        better use this version than the original one (from Oger).
    
    """
    pass

## RLS ##

class OnlineLinearRegression(mdp.Node):
    """Compute online least-square, multivariate linear regression on
    the input data, i.e., learn coefficients ``b_j`` so that::
    
      y_i = b_0 + b_1 x_1 + ... b_N x_N ,
    
    for ``i = 1 ... M``, minimizes the square error given the training
    ``x``'s and ``y``'s.
    
    This is a supervised learning node, and requires input data ``x``
    and target data ``y``. The training will never stop, the regression
    updated with every presented sample. Training can either be achieved
    by invoking ``train`` or also ``execute`` with additionally the
    targets given.
    
    **Internal variables of interest**
    
      ``self.beta``
          The coefficients of the linear regression
    
    .. note::
        
        There's been some issues when using the class within ESN_ACD and
        successors. It's currently adviced to use
        :py:class:`StabilizedRLS` instead. Main issues are using the
        class without having it trained once (leads to a training error)
        and the interface which requires a target output instead of an
        error term.
    
    """
    def __init__(self, with_bias=True, input_dim=None,
                output_dim=None, lambda_=1.0, dtype=None):
        """
        :Arguments:

          with_bias
            If true, the linear model includes a constant term

            - True:  y_i = b_0 + b_1 x_1 + ... b_N x_N
            - False: y_i =       b_1 x_1 + ... b_N x_N

            If present, the constant term is stored in the first
            column of ``self.beta``.
        """
        super(OnlineLinearRegression, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dtype=dtype
            )

        # final regression coefficients
        self.lambda_ = lambda_
        self.with_bias = with_bias
        self._is_initialized = False
    
    def _initialize(self, input_dim):
        if self._is_initialized: return
        
        if self.with_bias:
            input_dim += 1
            
        self.beta = np.zeros((input_dim, self.get_output_dim()))
        self._psiInv = np.eye(input_dim, input_dim) * 10000.0

        self._is_initialized = True

    @staticmethod
    def is_invertible():
        return False

    def _check_train_args(self, x, y):
        # set output_dim if necessary
        if self._output_dim is None:
            self._set_output_dim(y.shape[1])
        # check output dimensionality
        self._check_output(y)
        if y.shape[0] != x.shape[0]:
            msg = ("The number of output points should be equal to the "
                   "number of datapoints (%d != %d)" % (y.shape[0], x.shape[0]))
            raise TrainingException(msg)

    def _train(self, x, d):
        """Train the linear regression on some samples.
        
        ``x``
          array of size (K, input_dim) that contains the observed samples
        
        ``d``
          array of size (x.shape[0], output_dim) that contains the observed
          output to the input x's.
        
        """
        # initialize internal vars if necessary
        if not self._is_initialized: self._initialize(self.get_input_dim())
        if self.with_bias:
            x = self._add_constant(x)
        
        # update internal variables
        for n in range(x.shape[0]):
            xn = np.atleast_2d(x[n]).T
            dn = np.atleast_2d(d[n]).T
            u = self._psiInv.dot(xn)
            k = 1.0 / (self.lambda_ + xn.T.dot(u)) * u
            y = self.beta.T.dot(xn)
            e = dn - y
            self.beta += k.dot(e.T)
            self._psiInv -= k.dot(xn.T.dot(self._psiInv))
            self._psiInv /= self.lambda_
    
    def _stop_training(self):
        self._training = True

    def _execute(self, x, y=None):
        """
        
        """
        if self.with_bias:
            x = self._add_constant(x)
        
        if y is not None:
            # execute a training step
            self._train(x, y)
        
        return self.beta.T.dot(x.T).T

    def _add_constant(self, x):
        """Add a constant term to the vector 'x'.
        x -> [1 x]
        """
        return np.concatenate((np.ones((x.shape[0], 1), dtype=self.dtype), x), axis=1)

class PlainRLS:
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
    
    """
    def __init__(self, input_dim, output_dim, with_bias=True, lambda_=1.0):
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.lambda_ = lambda_
        self.with_bias = with_bias
        if self.with_bias: input_dim += 1
        self.beta = np.zeros((input_dim, self.output_dim))
        self._psiInv = np.eye(input_dim, input_dim) * 10000.0

    def train(self, x, d=None, e=None):
        """Train the regression on one or more samples.
        
        ``x``
            Input samples. Array of size (K, input_dim)
        
        ``d``
            Sample target. Array of size (K, output_dim)
        
        ``e``
            Sample error terms. Array of size (K, output_dim)
        
        """
        if self.with_bias: x = self._add_constant(x)
        for n in range(x.shape[0]):
            # preliminaries
            xn = np.atleast_2d(x[n]).T
            u = self._psiInv.dot(xn)
            k = 1.0 / (self.lambda_ + xn.T.dot(u)) * u
            # error
            if e is None:
                dn = np.atleast_2d(d[n]).T
                y = self.beta.T.dot(xn)
                en = dn - y
            else:
                en = np.atleast_2d(e[n]).T
            # update
            self.beta += k.dot(en.T)
            self._psiInv -= k.dot(xn.T.dot(self._psiInv))
            self._psiInv /= self.lambda_
    
    def __call__(self, x):
        """Evaluate the linear approximation on some point ``x``.
        """
        if self.with_bias: x = self._add_constant(x)
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
        >>> (r2._psiInv == r1._psiInv).all()
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
        import cPickle as pickle
        f = open(pth, 'w')
        pickle.dump(self, f)
        f.close()
    
    def __repr__(self):
        return 'PlainRLS(with_bias=%r, input_dim=%i, output_dim=%i, lambda_=%f)' % (self.with_bias, self.input_dim, self.output_dim, self.lambda_)

class StabilizedRLS(PlainRLS):
    """Compute online least-square, multivariate linear regression.
    
    Identical to :py:class:`PlainRLS`, except that the internal matrices
    are computed on the lower triangular part. This ensures symmetrical
    matrices even with floating point operations. If unsure, use this
    implementation instead of :py:class:`PlainRLS`.
    
    """
    def train(self, x, d=None, e=None):
        """Train the regression on one or more samples.
        
        ``x``
            Input samples. Array of size (K, input_dim)
        
        ``d``
            Sample target. Array of size (K, output_dim)
        
        ``e``
            Sample error terms. Array of size (K, output_dim)
        
        """
        if self.with_bias: x = self._add_constant(x)
        for n in range(x.shape[0]):
            # preliminaries
            xn = np.atleast_2d(x[n]).T
            u = self._psiInv.dot(xn)
            k = 1.0 / (self.lambda_ + xn.T.dot(u)) * u
            # error
            if e is None:
                dn = np.atleast_2d(d[n]).T
                y = self.beta.T.dot(xn)
                en = dn - y
            else:
                en = np.atleast_2d(e[n]).T
            # update
            self.beta += k.dot(en.T)
            tri = np.tril(self._psiInv)
            tri -= np.tril(k*u.T)
            tri /= self.lambda_
            self._psiInv = tri + tri.T - np.diag(tri.diagonal())
    
    def __repr__(self):
        return 'StabilizedRLS(with_bias=%r, input_dim=%i, output_dim=%i, lambda_=%f)' % (self.with_bias, self.input_dim, self.output_dim, self.lambda_)

