include("probefns.jl")


if abspath(PROGRAM_FILE) == @__FILE__
    graph_generator_name, V, start_train, n_train_interval, end_train, results_dir, run_index = ARGS
    V = parse(Int, V)
    K = ceil(V / 2) |> Int
    start_train = parse(Int, start_train)
    n_train_interval = parse(Int, n_train_interval)
    end_train = parse(Int, end_train)
    train_range = start_train:n_train_interval:end_train
    num_test = 1000
    graph_generator = getfield(Main, Symbol(graph_generator_name))
    # graph_generator = paley_graph
    # graph_generator = random_clique
    # graph_generator = random_bipartite
    println("Training on $(train_range) samples of $(graph_generator_name) at V = $(V), trial $(run_index).")
    optimiser1 = Optim.AcceleratedGradientDescent
    optimiser2 = Optim.GradientDescent
    optimiser3 = Optim.LBFGS
    # For Delta's clique
    optimiser4 = ()->Optim.MomentumGradientDescent(;mu=0.001)
    # For small Delta
    optimiser5 = Optim.Adam

    # For bipartite and paley
    if contains(["random_bipartite", "paley_graph"], graph_generator_name)
        make_obj_fns = Dict(
            "MEF"=>(fit_mef_2nd_order, optimiser1),
            "Delta"=>(fit_delta, optimiser1),
            "Perceptron"=>(fit_perceptron, optimiser2)  # actually doesn't matter
        )
    else
        make_obj_fns = Dict(
            "MEF"=>(fit_mef_2nd_order, optimiser1),
            "Delta"=>(fit_delta, optimiser5),
            "Perceptron"=>(fit_perceptron, optimiser2)
        )
    end

    # The result is an array of tuples (train, test, res) where res is a Dict of obj_name=>(net, optim_res, bitcorr%, corr%, correct_samples)
    if graph_generator_name == "random_group_graph"
        result = probe_train_range(V, K, train_range, num_test, graph_generator, make_obj_fns)
    else
        result = probe_iso_train_range(V, K, train_range, num_test, graph_generator, make_obj_fns)
    end
    # save results
    save_probe_result("$(results_dir)/probe_V=$(V)_K=$(K)_$(graph_generator_name)_$(run_index).jld2", result)
    plot_res = pack_plot_variables(result, keys(make_obj_fns) |> collect)
    save_probe_plot_result("$(results_dir)/probe_V=$(V)_K=$(K)_$(graph_generator_name)_$(run_index)_plot.jld2", plot_res)
    println("Finished trial $(run_index) of $(graph_generator_name) at V=$(V), K=$(K).")
end
