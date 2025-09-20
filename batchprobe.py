import os
import sys
from subprocess import run


def run_probe(trial_range, num_thread, run_type, V, start_sample, interval_sample, end_sample):
    """
    For running runprobe.jl in batches.
    """
    result_loc = f"results/probe/{run_type}/"
    for t in trial_range:
        cmd = ["julia", "-t", f"{num_thread}", "runprobe.jl", f"{run_type}", f"{V}", f"{start_sample}", f"{interval_sample}", f"{end_sample}", f"{result_loc}", f"{t}"]
        run(cmd)

if __name__ == "__main__":
    V, start_sample, interval_sample, end_sample, start_trial, num_trial = [int(i) for i in sys.argv[1:]]
    trial_range = range(start_trial, start_trial+num_trial)
    #run_types = ["random_bipartite", "paley_graph"]
    run_types = ["random_clique"]
    #run_types = ["random_group_graph"]
    num_thread = os.cpu_count()
    for run_type in run_types:
        run_probe(trial_range, num_thread, run_type, V, start_sample, interval_sample, end_sample)
