"""
Reservoir Computing code
"""

import Oger
import mdp
import warnings
import numpy as np

class SparseReservoirNode(Oger.nodes.ReservoirNode):
    """
    Oger.nodes.SparseReservoirNode with some corrections and extensions:
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
        """
        Oger.nodes.SparseReservoirNode.sparse_w with density correction.
        Now, the density is exactly as set by *fan_in_w*, given in
        percent (0-100).
        
        """
        import scipy.sparse
        from scipy.sparse.linalg import eigs
        converged = False
        # Initialize reservoir weight matrix
        nrentries = int((out_size**2 * self.fan_in_w)/100.0)
        out_size = int(out_size)
        # Keep generating random matrices until convergence
        while not converged:
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
        return w
    
    def dense_w_in(self, out_size, in_size, scaling, rnd_fu=lambda size: np.random.uniform(size=size)*2.0-1.0):
        return scaling * rnd_fu(size=(self.output_dim, self.input_dim))
    
    def sparse_w_in(self, out_size, in_size, scaling, rnd_fu=lambda size: np.random.uniform(size=size)*2.0-1.0):
        """
        Oger.nodes.SparseReservoirNode.sparse_w_in with density
        correction. Now, the density is exactly as defined by
        *fan_in_i*, given in percent (0-100). The values are drawn
        from an uniform distribution.
        
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
        self.states = mdp.numx.zeros((1, self.output_dim))
    
    def _execute(self, x, simulate=False):
        """Executes simulation with input vector x.
        
        ``x``
            Input samples. Variables in columns, observations in rows,
            i.e. if N,M = x.shape then M == *input_dim*.
        
        ``simulate``
            If *True*, the state won't be updated.
        
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

        # Save the state for re-initialization in case reset_states = False
        if not simulate:
            self.states = states[1:, :]

        # Return the whole state matrix except the initial state
        return self.states

class OrthogonalReservoirNode(SparseReservoirNode):
    def sparse_w(self, out_size, specrad, rnd_fu=np.random.normal):
        """
        Orthogonal reservoir construction algorithm
        """
        import scipy.sparse
        from scipy.sparse.linalg import eigs

class FreerunFlow(Oger.nodes.FreerunFlow):
    """Oger.nodes.FreerunFlow with corrections and extensions
    
    .. todo::
        The class should be usable within the cross-validation framework
        
    """
    pass


class OnlineLinearRegression(mdp.Node):
    """Compute online least-square, multivariate linear regression on the input
    data, i.e., learn coefficients ``b_j`` so that::

      y_i = b_0 + b_1 x_1 + ... b_N x_N ,

    for ``i = 1 ... M``, minimizes the square error given the training ``x``'s
    and ``y``'s.

    This is a supervised learning node, and requires input data ``x`` and
    target data ``y``. The training will never stop, the regression updated
    with every presented sample. Training can either be achieved by invoking
    ``train`` or also ``execute`` with additionally the targets given.

    **Internal variables of interest**

      ``self.beta``
          The coefficients of the linear regression
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
        """
        **Additional input arguments**
        x
          array of size (K, input_dim) that contains the observed samples
        y
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
