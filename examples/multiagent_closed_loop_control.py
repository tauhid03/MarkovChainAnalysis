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
        f.write(vi.policy)



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
    print(states)

    while not checkEqual(states) and steps <= max_steps:
        next_states = run_policy(N, states, Ps)
        steps += 1
        states = next_states

    return steps, states


@click.command()
@click.option('--num_agents', '-n', default=2)
def execute_policy(num_agents):
    X = len(env2)
    Ps, R = mkSimpleRendezvousMDP(env2, num_agents, types2)
    with open("policy"+str(num_agents)+".txt", 'r') as f:
        str_policy = f.read().strip('()\n').split(', ')

    policy = [int(x) for x in str_policy]
    N_trials = 100
    start_states = np.random.choice(X, N_trials)
    steps = [0]*N_trials
    for i,s in enumerate(start_states):
        steps[i], sts = steps_until_rendezvous(num_agents, s, Ps, policy, X)

def run_MDP(num_agents):
        example_one_obstacle_6_states(num_agents)

if __name__ == '__main__':
    #run_MDP()
    execute_policy()
