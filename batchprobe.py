import os
import sys
from subprocess import run


def run_probe(subrun_name, trial_range, num_thread, models, run_type, V, start_sample, interval_sample, end_sample):
    """
    For running runprobe.jl in batches.
    """
    result_loc = f"results/{subrun_name}/{run_type}/"
    for t in trial_range:
        cmd = ["julia", "-t", f"{num_thread}", "runprobe.jl", f"{models}", f"{run_type}", f"{V}", f"{start_sample}", f"{interval_sample}", f"{end_sample}", f"{result_loc}", f"{t}"]
        run(cmd)

if __name__ == "__main__":
    subrun_name, models, run_types = sys.argv[1:4]
    V, start_sample, interval_sample, end_sample, start_trial, num_trial = [int(i) for i in sys.argv[4:]]
    trial_range = range(start_trial, start_trial+num_trial)
    run_types = run_types.split(",")
    assert all(r in ["random_bipartite", "paley_graph", "random_clique", "random_group_graph"] for r in run_types)
    mtypes = models.split(",")
    assert all(m in ["MEF", "Delta", "Perceptron"] for m in mtypes)
    num_thread = os.cpu_count()
    for run_type in run_types:
        run_probe(subrun_name, trial_range, num_thread, models, run_type, V, start_sample, interval_sample, end_sample)
