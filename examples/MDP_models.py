import numpy as _np
from random import random
from copy import deepcopy

'''
define topology for 4x5 rectangular environment
see diagram (o = free, x = obstacle)
o x o o o
o x o o o
o o o x o
o o o x o
'''

env1 =  {
    0:  {                4:'S'         },
    1:  {        2:'E',  5:'S',        },
    2:  {        3:'E',  6:'S',  1:'W' },
    3:  {                7:'S',  2:'W' },
    4:  {0:'N',          8:'S',        },
    5:  {1:'N',  6:'E',  10:'S'        },
    6:  {2:'N',  7:'E',          5:'W' },
    7:  {3:'N',          11:'S', 6:'W' },
    8:  {4:'N',  9:'E',  12:'S'        },
    9:  {        10:'E', 13:'S', 8:'W' },
    10: {5:'N',          14:'S', 9:'W' },
    11: {7:'N',          15:'S'        },
    12: {8:'N',  13:'E'                },
    13: {9:'N',  14:'E',         12:'W'},
    14: {10:'N',                 13:'W'},
    15: {11:'N'                        }
    }

'''
see diagram (number = id of state, x = obstacle)
0 0 1 2 3
0 0 4 x 5
0 0 4 x 5
'''

env2 =  {
    0:  {        1:'E', 4:'E'          },
    1:  {        2:'E', 4:'S',   0:'W' },
    2:  {        3:'E',          1:'W' },
    3:  {5:'S', 2:'W'                  },
    4:  {1:'N', 0:'W'                  },
    5:  {3:'N'                         },
    }

dirs = {0:'N', 1:'E', 2:'S', 3:'W'}

def neighbors(state, env):
        return list(env[state].keys())


'''
TODO: simple function checking bidirectionality and grid toplogy
'''
def check_grid_topo(env):
    symm_all = True
    for s in env.keys():
        symm_ns = True
        for n in neighbors(s, env):
            symm_ns = symm_ns and (s in neighbors(n, env))
        symm_all = symm_all and symm_ns
    return symm_all


'''
For each type, alpha is the probability that the agent continues its motion in
the current direction.

Therefore, 1-alpha is the probability of a "tumble" event, where the orientation
switches. Here, we implement a uniformly random probability of switching to
another direction.
'''

type_A = {
    'alpha': 0.8, # probability of continuing in current dir
    'beta': 0.2 # probability of staying in current gridcell
    }

type_B = {
    'alpha': 0.2,
    'beta': 0.95
}


# TODO: make prob a function of size of state

# uniform random walk
type_rw = {
    'p_stay':0.2,
    'p_N':0.2,
    'p_E':0.2,
    'p_S':0.2,
    'p_W':0.2
    }

# low prob of moving north or staying in place
type_dir = {
    'p_stay':0.25,
    'p_N':0.05,
    'p_E': 0.3,
    'p_S':0.3,
    'p_W': 0.3
    }

types1 = [type_A, type_B]

types2 = [type_rw, type_dir]

# function to compute each entry in the transition matrix
def gridworld_step_prob(curr_state, next_state, type, env):
    ns = neighbors(curr_state, env)

    if curr_state == next_state:
        return type["p_stay"]
    elif next_state not in ns:
        return 0.0
    else:
        for n in ns:
            if next_state == n:
                dir = env[curr_state][next_state]
                key = "p_"+dir
                return type[key]


'''
input: list of individual robot states (encoded format)
output: encoded index of joint system state in base S

'''
def encodeJointState(states, X):
    X = _np.uint64(X)
    joint_state = _np.uint64(0)
    N = _np.uint64(states.size-1)
    for s in states:
        joint_state += _np.uint64(s*(X**N))
        N = N - 1
    return joint_state

def decodeJointState(state, N, X):
    state = _np.uint64(state)
    X = _np.uint64(X)
    states = _np.empty(N, dtype= _np.uint64)
    for n in range(N):
        next_state = state % X
        states[n] = next_state
        state = state // X
    return states[::-1]


def mkTransitions(env, types, N, X, transition):
    Ps = _np.zeros((2,N,X,X))
    for (i,type) in enumerate(types):
        for j in range(N):
            temp_P = _np.zeros((X, X))
            for start in range(X):
                for end in range(X):
                    temp_P[start][end] =  transition(start, end, type, env)

            # normalize rows of transition matrix, just in case
            Ps[i][j] = temp_P/_np.linalg.norm(temp_P, ord=1, axis=1, keepdims=True)
    return Ps

'''
Define a reward vector which rewards all agents being in the same grid cell,
regardless of orientation.

Assumes grid cells are much larger than agents.
'''
def mkRendezvousReward(N, X):
    S = _np.uint64(X**N)
    R = _np.full(S, -1.0)
    for state in range(X):
        states = _np.full(N, state, dtype= _np.uint64)
        joint_state = encodeJointState(states, X)
        R[joint_state] = 1.0
    return R

def mkSimpleRendezvousMDP(env, N, types=types2):
    X = len(env)
    Ps = mkTransitions(env, types, N, X, gridworld_step_prob)
    R = mkRendezvousReward(N, X)
    return Ps, R

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
    return (Ps, R)

def multiagent_full(S=10, N=4, pslow = 0.9, pfast = 0.2):
    Ps, R = multiagent(S, N, pslow, pfast)
    Ps0 = [P for P in Ps[0]]
    Ps1 = [P for P in Ps[1]]
    P0 = reduce(lambda x, y: _np.dot(x,y), _np.identity(S**N), Ps0)
    P1 = reduce(lambda x, y: _np.dot(x,y), _np.identity(S**N), Ps1)
    return [P0, P1], R
