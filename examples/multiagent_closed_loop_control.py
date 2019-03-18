from src.KronMDP import KronValueIteration
from examples.MDP_models import env1, mkRendezvousMDP, types1
from timeit import default_timer as timer
from functools import reduce


def example_two_obstacles(N):
    Ps, R = mkRendezvousMDP(env1, types1, N)
    start = timer()
    vi = KronValueIteration(Ps, R, 0.95, skip_check=True, sparse=True)
    vi.run()
    end = timer()
    print("kronecker method took", end-start,"seconds")
    print(vi.policy)

if __name__ == "__main__":
    example_two_obstacles(4)
