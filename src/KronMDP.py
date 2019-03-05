#!/usr/bin/env python

import numpy as _np
import time as _time
from mdptoolbox.mdp import MDP, _printVerbosity, _MSG_STOP_EPSILON_OPTIMAL_POLICY, _MSG_STOP_MAX_ITER
import mdptoolbox.util as _util
from src.kronprod import KronProd
from functools import reduce

# returns Ps as an array, indexed first by action and then by subsystem
# action 0 makes agents slow (tend to stay in same state)
# action 1 make agents fast (tend to move to one of the neighboring states)
def multiagent(S=10, N=4, pslow = 0.9, pfast = 0.2):
    """Generate a MDP example for N random agents, each with state space size S
       Assume two actions: fast or slow
    """
    assert S > 1, "The number of states S must be greater than 1."
    assert N > 1, "The number of agents N must be greater than 1."
    # Definition of Transition matrices
    Ps = _np.zeros((2,N,S,S))
    for i in range(N):
        Ps[0][i] = _np.zeros((S, S))
        Ps[0][i][:, :] += (1 - pslow)/2 * _np.diag(_np.ones(S - 1), 1)
        Ps[0][i][:, :] += (1 - pslow)/2 * _np.diag(_np.ones(S - 1), -1)
        Ps[0][i][0, -1] += (1 - pslow)/2
        Ps[0][i][-1, 0] += (1 - pslow)/2
        Ps[0][i][:, :] += pslow * _np.diag(_np.ones(S))

        Ps[1][i] = _np.zeros((S, S))
        Ps[1][i][:, :] += (1 - pfast)/2 * _np.diag(_np.ones(S - 1), 1)
        Ps[1][i][:, :] += (1 - pfast)/2 * _np.diag(_np.ones(S - 1), -1)
        Ps[1][i][:, :] += pfast * _np.diag(_np.ones(S))
        Ps[1][i][0, -1] += (1 - pfast)/2
        Ps[1][i][-1, 0] += (1 - pfast)/2
    # Definition of Reward matrix
    # Reward for both agents is zero everywhere but in last state
    R = _np.zeros((S**N, 2))
    R[-1, 0] = 1
    R[:, 1] = _np.zeros(S**N)
    R[0, 1] = 0
    R[S - 1, 1] = 1
    return(Ps, R)

def multiagent_full(S=10, N=4, pslow = 0.9, pfast = 0.2):
    Ps, R = multiagent(S, N, pslow, pfast)
    Ps0 = [P for P in Ps[0]]
    Ps1 = [P for P in Ps[1]]
    P0 = reduce(lambda x, y: _np.dot(x,y), _np.identity(S**N), Ps0)
    P1 = reduce(lambda x, y: _np.dot(x,y), _np.identity(S**N), Ps1)
    return [P0, P1], R

# for kron problems, transition is a list of lists of arrays
# first indexed by subsystem, then action
class KronMDP(MDP):

    """A Markov Decision Problem, specialized for transition matrices
    represented as factored Kroenicker product.

    Let ``S`` = the number of states in one of the matrices in the Kroenicker
    product, and ``A`` = the number of acions, and
    ``N`` = the number of "agents" or factors in the Kroenicker product.
    (Thus the overall state space of the problem will be size ``S^N``)

    Parameters
    ----------
    transitions : array
        Transition matrices, with shape ``(A, N, S, S)``.
    reward : array
        Reward matrices or vectors. Like the transition matrices, these can
        also be defined in a variety of ways. Again the simplest is a numpy
        array that has the shape ``(S, A)``, ``(S,)`` or ``(A, S, S)``. A list
        of lists can be used, where each inner list has length ``S`` and the
        outer list has length ``A``. A list of numpy arrays is possible where
        each inner array can be of the shape ``(S,)``, ``(S, 1)``, ``(1, S)``
        or ``(S, S)``. Also ``scipy.sparse.csr_matrix`` can be used instead of
        numpy arrays. In addition, the outer list can be replaced by any object
        that can be indexed like ``reward[a]`` such as a tuple or numpy object
        array of length ``A``.
    discount : float
        Discount factor. The per time-step discount factor on future rewards.
        Valid values are greater than 0 upto and including 1. If the discount
        factor is 1, then convergence is cannot be assumed and a warning will
        be displayed. Subclasses of ``MDP`` may pass ``None`` in the case where
        the algorithm does not use a discount factor.
    epsilon : float
        Stopping criterion. The maximum change in the value function at each
        iteration is compared against ``epsilon``. Once the change falls below
        this value, then the value function is considered to have converged to
        the optimal value function. Subclasses of ``MDP`` may pass ``None`` in
        the case where the algorithm does not use an epsilon-optimal stopping
        criterion.
    max_iter : int
        Maximum number of iterations. The algorithm will be terminated once
        this many iterations have elapsed. This must be greater than 0 if
        specified. Subclasses of ``MDP`` may pass ``None`` in the case where
        the algorithm does not use a maximum number of iterations.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.

    Attributes
    ----------
    P : array
        Transition probability matrices.
    R : array
        Reward vectors.
    V : tuple
        The optimal value function. Each element is a float corresponding to
        the expected value of being in that state assuming the optimal policy
        is followed.
    discount : float
        The discount rate on future rewards.
    max_iter : int
        The maximum number of iterations.
    policy : tuple
        The optimal policy.
    time : float
        The time used to converge to the optimal policy.
    verbose : boolean
        Whether verbose output should be displayed or not.

    Methods
    -------
    run
        Implemented in child classes as the main algorithm loop. Raises an
        exception if it has not been overridden.
    setSilent
        Turn the verbosity off
    setVerbose
        Turn the verbosity on

    """

    def __init__(self, transitions, reward, discount, epsilon, max_iter,
                 skip_check=True):
        # Initialise a MDP based on the input parameters.

        # if the discount is None then the algorithm is assumed to not use it
        # in its computations
        if discount is not None:
            self.discount = float(discount)
            assert 0.0 < self.discount <= 1.0, (
                "Discount rate must be in ]0; 1]"
            )
            if self.discount == 1:
                print("WARNING: check conditions of convergence. With no "
                      "discount, convergence can not be assumed.")

        # if the max_iter is None then the algorithm is assumed to not use it
        # in its computations
        if max_iter is not None:
            self.max_iter = int(max_iter)
            assert self.max_iter > 0, (
                "The maximum number of iterations must be greater than 0."
            )

        # check that epsilon is something sane
        if epsilon is not None:
            self.epsilon = float(epsilon)
            assert self.epsilon > 0, "Epsilon must be greater than 0."

        # this will fail for Kroneicker representation right now - but we can
        # write a new check function to make sure dimensions match
        if not skip_check:
            # We run a check on P and R to make sure they are describing an
            # MDP. If an exception isn't raised then they are assumed to be
            # correct.
            _util.check(transitions, reward)

        self.A = transitions.shape[0]
        self.P = self._computeTransition(transitions)
        self.S = self.P[0].N
        self.R = self._computeReward(reward, transitions)

        # the verbosity is by default turned off
        self.verbose = False
        # Initially the time taken to perform the computations is set to None
        self.time = None
        # set the initial iteration count to zero
        self.iter = 0
        # V should be stored as a vector ie shape of (S,) or (1, S)
        self.V = None
        # policy can also be stored as a vector
        self.policy = None

    def __repr__(self):
        P_repr = "P: \n"
        R_repr = "R: \n"
        for aa in range(self.A):
            P_repr += repr(self.P[aa]) + "\n"
            R_repr += repr(self.R[aa]) + "\n"
        return(P_repr + "\n" + R_repr)

    def _computeTransition(self, transition):
        return tuple(KronProd(transition[a]) for a in range(self.A))

    def _bellmanOperator(self, V=None):
        # Apply the Bellman operator on the value function.
        #
        # Updates the value function and the Vprev-improving policy.
        #
        # Returns: (policy, value), tuple of new policy and its value
        #
        # If V hasn't been sent into the method, then we assume to be working
        # on the objects V attribute
        if V is None:
            # this V should be a reference to the data rather than a copy
            V = self.V
        else:
            # make sure the user supplied V is of the right shape
            try:
                assert V.shape in ((self.S,), (1, self.S)), "V is not the " \
                    "right shape (Bellman operator)."
            except AttributeError:
                raise TypeError("V must be a numpy array or matrix.")
        # Looping through each action the the Q-value matrix is calculated.
        # P and V can be any object that supports indexing, so it is important
        # that you know they define a valid MDP before calling the
        # _bellmanOperator method. Otherwise the results will be meaningless.
        Q = _np.empty((self.A, self.S))
        for aa in range(self.A):
            Q[aa] = self.R[aa] + self.discount * self.P[aa].dot(V)
        # Get the policy and value, for now it is being returned but...
        # Which way is better?
        # 1. Return, (policy, value)
        return (Q.argmax(axis=0), Q.max(axis=0))
        # 2. update self.policy and self.V directly
        # self.V = Q.max(axis=1)
        # self.policy = Q.argmax(axis=1)

    def _startRun(self):
        if self.verbose:
            _printVerbosity('Iteration', 'Variation')

        self.time = _time.time()

    def _endRun(self):
        # store value and policy as tuples
        self.V = tuple(self.V.tolist())

        try:
            self.policy = tuple(self.policy.tolist())
        except AttributeError:
            self.policy = tuple(self.policy)

        self.time = _time.time() - self.time

    def run(self):
        """Raises error because child classes should implement this function.
        """
        raise NotImplementedError("You should create a run() method.")

    def setSilent(self):
        """Set the MDP algorithm to silent mode."""
        self.verbose = False

    def setVerbose(self):
        """Set the MDP algorithm to verbose mode."""
        self.verbose = True

class KronValueIteration(KronMDP):
        
    """A discounted MDP solved using the value iteration algorithm, with
    transition matrix represented as a Kroenicker product

    Description
    -----------

    Parameters
    ----------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    discount : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    epsilon : float, optional
        Stopping criterion. See the documentation for the ``MDP`` class for
        details.  Default: 0.01.
    max_iter : int, optional
        Maximum number of iterations. If the value given is greater than a
        computed bound, a warning informs that the computed bound will be used
        instead. By default, if ``discount`` is not equal to 1, a bound for
        ``max_iter`` is computed, otherwise ``max_iter`` = 1000. See the
        documentation for the ``MDP`` class for further details.
    initial_value : array, optional
        The starting value function. Default: a vector of zeros.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.

    Data Attributes
    ---------------
    V : tuple
        The optimal value function.
    policy : tuple
        The optimal policy function. Each element is an integer corresponding
        to an action which maximises the value function in that state.
    iter : int
        The number of iterations taken to complete the computation.
    time : float
        The amount of CPU time used to run the algorithm.

    Methods
    -------
    run()
        Do the algorithm iteration.
    setSilent()
        Sets the instance to silent mode.
    setVerbose()
        Sets the instance to verbose mode.

    Notes
    -----
    In verbose mode, at each iteration, displays the variation of V
    and the condition which stopped iterations: epsilon-optimum policy found
    or maximum number of iterations reached.

    Examples
    --------

    """

    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=1000, initial_value=0, skip_check=False):
        # Initialise a value iteration MDP.

        KronMDP.__init__(self, transitions, reward, discount, epsilon, max_iter,
                     skip_check=skip_check)

        # initialization of optional arguments
        if initial_value == 0:
            self.V = _np.zeros(self.S)
        else:
            assert len(initial_value) == self.S, "The initial value must be " \
                "a vector of length S."
            self.V = _np.array(initial_value).reshape(self.S)
        if self.discount < 1:
            # compute a bound for the number of iterations and update the
            # stored value of self.max_iter
            # TODO adapt this bound for kronprod case: can it be computed
            # efficiently?
            #self._boundIter(epsilon)
            # computation of threshold of variation for V for an epsilon-
            # optimal policy
            self.thresh = epsilon * (1 - self.discount) / self.discount
        else:  # discount == 1
            # threshold of variation for V for an epsilon-optimal policy
            self.thresh = epsilon

    def run(self):
        self.verbose = True
        # Run the value iteration algorithm.
        self._startRun()

        while True:
            self.iter += 1

            Vprev = self.V.copy()

            # Bellman Operator: compute policy and value functions
            self.policy, self.V = self._bellmanOperator()

            # The values, based on Q. For the function "max()": the option
            # "axis" means the axis along which to operate. In this case it
            # finds the maximum of the the rows. (Operates along the columns?)
            variation = _util.getSpan(self.V - Vprev)

            if self.verbose:
                _printVerbosity(self.iter, variation)

            if variation < self.thresh:
                if self.verbose:
                    print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                break
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break

        self._endRun()


