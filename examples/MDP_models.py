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


def gridworld_step_prob_w_dirs(curr_state, next_state, type, env):
    curr_gridcell, curr_dir = decode_state(curr_state, env)
    next_gridcell, next_dir = decode_state(next_state, env)
    ns = neighbors(curr_gridcell, env)
    if (curr_gridcell == next_gridcell):
        return type['beta']
    elif next_gridcell not in ns:
        return 0.0
    else:
        if dirs[curr_dir] == env[curr_gridcell][next_gridcell]:
            return (1.0-type['beta'])*type['alpha']
        else:
            return (1.0-type['beta'])*(1.0 - type['alpha'])/(len(ns)+1) # +1 because of prob of staying in current state


'''
The encoding scheme I use for states is to assign a unique integer, n, to each grid
cell, in row-column order. Let M be the number of grid cells.

There are four orientations, also assigned integers, N:=0, E:=1, S:=2, W:=3. Let
O be the total number of orientations

States are then indexed n + M*O

'''

def encode_state(gridcell, dir, env):
    return gridcell + len(env)*dir


def decode_state(n, env):
    gridcell = n % len(env)
    dir = n // len(env)
    return gridcell, dir

'''
input: list of individual robot states (encoded format)
output: encoded index of joint system state in base S

'''
def encodeJointState(states, env, N, S):
    joint_state = 0
    for s in states:
        joint_state += s*(S**N)
        N = N - 1
    return joint_state

def decodeJointState(state, env, N, S):
    states = []
    for n in range(N):
        states.append(state % S)
        state = state // S
    return states[::-1]


def mkTransitions(env, types, N, S, transition):
    Ps = _np.zeros((2,N,S,S))
    for (i,type) in enumerate(types):
        for j in range(N):
            temp_P = _np.zeros((S, S))
            for start in range(S):
                for end in range(S):
                    temp_P[start][end] =  transition(start, end, type, env)

            # normalize rows of transition matrix, just in case
            Ps[i][j] = temp_P/_np.linalg.norm(temp_P, ord=1, axis=1, keepdims=True)
    return Ps

'''
Define a reward vector which rewards all agents being in the same grid cell,
regardless of orientation.

Assumes grid cells are much larger than agents.
'''
def mkRendezvousReward(env, N, S):

    X = S**N
    R = _np.full((S**N), -1.0)
    for cell in range(len(env)):
        for dir in range(4):
            state = encode_state(cell, dir, env)
            states = [state for robot in range(N)]
            R[encodeJointState(states, env, N-1, S)] = 1.0
    return R

def mkRendezvousMDP(env, N, types=types1):
    S = len(env)*4
    Ps = mkTransitions(env, types, N, S, gridworld_step_prob_w_dirs)
    R = mkRendezvousReward(env, N, S)
    return Ps, R


def mkSimpleRendezvousMDP(env, N, types=types2):
    S = len(env)
    Ps = mkTransitions(env, types, N, S, gridworld_step_prob)
    R = mkRendezvousReward(env, N, S)
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
