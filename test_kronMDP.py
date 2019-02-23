from mdptoolbox.example import rand
from mdptoolbox.mdp import ValueIteration
from src.KronMDP import multiagent, multiagent_full, KronValueIteration
from timeit import default_timer as timer
from functools import reduce

RUNBIG = False
RUNKRON = True
RUNFULL = True

# large example with memory problems - python cannot create example
if RUNBIG:
    P,R = rand(100000, 5)
    vi = ValueIteration(P,R,0.95)
    vi.run()

# kron example (not as dense)
if RUNKRON:
    Ps, R = multiagent(S=10, N=5)
    start = timer()
    vi = KronValueIteration(Ps, R, 0.95, skip_check=True)
    vi.run()
    end = timer()
    print("kronecker method took", end-start,"seconds")

# compare with fully computed example
if RUNFULL:
    P, R = multiagent_full(S=10, N=5)
    start = timer()
    vi = ValueIteration(P, R, 0.95)
    vi.run()
    end = timer()
    print("unfactored method took", end-start,"seconds")
