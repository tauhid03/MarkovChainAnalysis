import numpy as np
from timeit import default_timer as timer
from functools import reduce
import click
import sys

from src.KronMDP import KronValueIteration
from examples.MDP_models import *

def example_two_obstacles_64_states(N):
    Ps, R = mkRendezvousMDP(env1, N, types1)
    start = timer()
    vi = KronValueIteration(Ps, R, 0.95, skip_check=True, sparse=True)
    vi.run()
    end = timer()
    print("kronecker method took", end-start,"seconds")
    print(vi.policy)
    sys.stdout.flush()


def example_one_obstacle_6_states(N):
    Ps, R = mkSimpleRendezvousMDP(env2, N, types2)
    start = timer()
    vi = KronValueIteration(Ps, R, 0.95, skip_check=True, sparse=True)
    vi.run()
    end = timer()
    print("kronecker method took", end-start,"seconds")
    print(vi.policy)
    sys.stdout.flush()
    with open("policy"+str(N)+".txt", 'w') as f:
        f.write(str(vi.policy))



# choose value from normalized list of transition probabilities
# choose proportionally to value of transition probability
def choose_next_state(row):
    r = random()
    sum = 0.0
    for i,val in enumerate(row):
        sum += val
        if sum >= r:
            return i
    # return last element if we make it through the loop, just in case of weird
    # floating point things
    return row.size-1

# check if we are in a rendezvous state
# will short-circuit if it finds a non-matching state
def checkEqual(states):
    first = states[0]
    return all(x==first for x in states)

def run_policy(N, states, Ps):
    # step each agent forward, according to the appropriate
    # distribution
    for agent in range(N):
        P = Ps[agent]
        state = states[agent]
        states[agent] = choose_next_state(P[state])
    return states

def steps_until_rendezvous(N, state_N, allPs, policy, X):
    steps = 0
    max_steps = 200
    mdp_policy = policy[state_N]
    Ps = allPs[mdp_policy]
    states = decodeJointState(state_N, N, X)

    while not checkEqual(states) and steps <= max_steps:
        next_states = run_policy(N, states, Ps)
        steps += 1
        states = next_states

    return steps


def execute_policy(N, policy, start_states):
    X = len(env2)
    Ps, R = mkSimpleRendezvousMDP(env2, N, types2)
    steps = {s:0 for s in start_states}
    for s in start_states:
        steps[s] = steps_until_rendezvous(N, s, Ps, policy, X)
    return steps

def run_MDP(N):
        example_one_obstacle_6_states(N)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 1:
        N = np.uint64(args[0])
    else:
        N = 3
        print("Either none or too many arguments provided; please provide a single integer for number of agents")

    run_MDP(N)
    with open("policy"+str(N)+".txt", 'r') as f:
        str_policy = f.read().strip('()\n').split(', ')

    X = len(env2)
    S = np.uint64(X**N)
    N_trials = 100
    start_states = np.random.choice(S, N_trials)

    mdp_policy = [int(x) for x in str_policy]
    mdp_steps = execute_policy(N, mdp_policy, start_states)

    constant_policy_0 = [0 for x in mdp_policy]
    policy0_steps = execute_policy(N, constant_policy_0, start_states)

    constant_policy_1 = [1 for x in mdp_policy]
    policy1_steps = execute_policy(N, constant_policy_1, start_states)

    mdp_data = [n for j, n in mdp_steps.items()]
    p0_data = [n for j, n in policy0_steps.items()]
    p1_data = [n for j, n in policy1_steps.items()]

    with open("MDP_policy_runtimes_"+str(N)+"_agents.txt", 'a') as f:
        f.write("Average steps:"+str(np.mean(mdp_data)))
        f.write("STD steps:"+str(np.std(mdp_data)))
        for joint_state, num_steps in mdp_steps.items():
            f.write(str(joint_state)+ ": "+str(num_steps)+'\n')

    with open("constant_policy_0_runtimes_"+str(N)+"_agents.txt", 'a') as f:
        f.write("Average steps:"+str(np.mean(p0_data))+'\n')
        f.write("STD steps:"+str(np.std(p0_data))+'\n')
        for joint_state, num_steps in policy0_steps.items():
            f.write(str(joint_state)+ ": "+str(num_steps)+'\n')

    with open("constant_policy_1_runtimes_"+str(N)+"_agents.txt", 'a') as f:
        f.write("Average steps:"+str(np.mean(p1_data))+'\n')
        f.write("STD steps:"+str(np.std(p1_data))+'\n')
        for joint_state, num_steps in policy1_steps.items():
            f.write(str(joint_state)+ ": "+str(num_steps)+'\n')



