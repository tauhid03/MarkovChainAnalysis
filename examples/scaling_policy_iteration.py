from mdptoolbox.example import rand
from mdptoolbox.mdp import ValueIteration, PolicyIteration, _LP
from examples.MDP_models import multiagent, multiagent_full
from src.KronMDP import KronValueIteration, KronPolicyIteration
from timeit import default_timer as timer
from functools import reduce

RUNBIG = False
RUNKRON = True
RUNFULL = False

# large example with memory problems - python cannot create example
if RUNBIG:
    start = timer()
    P,R = rand(1000000, 5)
    vi = PolicyIteration(P,R,0.95)
    vi.run()
    end = timer()
    print("Full method took", end-start,"seconds")

# kron example (not as dense)
if RUNKRON:
    Ps, R = multiagent(S=10, N=5)
    start = timer()
    vi = KronPolicyIteration(Ps, R, 0.95, skip_check=True)
    vi.run()
    end = timer()
    print("kronecker method took", end-start,"seconds")

# compare with fully computed example
if RUNFULL:
    P, R = multiagent_full(S=10, N=3)
    start = timer()
    vi = KronPolicyIteration(P, R, 0.95)
    vi.run()
    end = timer()
    print("unfactored method took", end-start,"seconds")
