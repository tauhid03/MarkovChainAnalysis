from src.KronMDP import KronValueIteration
from examples.MDP_models import env1, mkRendezvousMDP, types1
from timeit import default_timer as timer
from functools import reduce
import click


def example_two_obstacles(N):
    Ps, R = mkRendezvousMDP(env1, types1, N)
    start = timer()
    vi = KronValueIteration(Ps, R, 0.95, skip_check=True, sparse=True)
    vi.run()
    end = timer()
    print("kronecker method took", end-start,"seconds")
    print(vi.policy)


@click.command()
@click.option('--num_agents', '-n', default=2)
def run_example(num_agents):
    example_two_obstacles(num_agents)

if __name__ == '__main__':
    run_example()
