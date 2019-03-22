import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

def plot_steps(step_data):
    print(step_data)

    df = pd.DataFrame(step_data, columns=["Policy","N", "steps"])
    print(df)

    plot = sns.boxplot(data=df, x="N", y="steps", hue="Policy")
    plot.get_figure().savefig("trend.png")

def read_file(N, tag):
    steps = []
    with open(tag+"_policy_runtimes_"+str(N)+"_agents.txt", 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            [start_state, stepN] = line.strip('\n').split(': ')
            steps.append((tag, N, int(stepN)))
    return steps


def read_data(Ns):
    step_data = []
    for i,N in enumerate(Ns):
        tags = ["MDP", "const0", "const1"]
        for t in tags:
            try:
                print("reading",t)
                step_data.extend(read_file(N, t))
            except FileNotFoundError:
                pass
    return step_data

if __name__ == '__main__':
    step_data = read_data([2,3,4])
    plot_steps(step_data)
