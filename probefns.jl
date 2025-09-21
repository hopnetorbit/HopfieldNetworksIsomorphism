include("hopfield.jl")
import JLD2
import Plots
import IterTools
import Random


function save_probe_plot(save_path::String, title::String, plot_res::Union{Dict, JLD2.JLDFile})
    colours = [:blue :green :red :cyan :yellow :purple][1:length(plot_res)]
    colours = zip(colours, colours) |> Iterators.flatten |> collect
    colours = reshape(colours, 1, length(colours))
    line_temp = [:dashdot :solid]
    linestyles = Matrix(undef, 1, 0)
    labels = []
    ys = []
    local x
    for obj_name in keys(plot_res)
        res = plot_res[obj_name]
        push!(ys, res["bitcorr"])
        push!(ys, res["corr"])
        linestyles = hcat(linestyles, line_temp)
        push!(labels, "$(obj_name) bit")
        push!(labels, "$(obj_name)")
        x = res["x"]
    end
    labels = reshape(labels, 1, length(labels))
    Plots.plot(x, hcat(ys...), label=labels, color=colours, linestyle=linestyles, title=title, legend=:best)
    Plots.savefig(save_path)
end


function save_probe_result(fpath::String, result::AbstractVector)
    if isfile(fpath)
        println("Joining result")
        f = JLD2.jldopen(fpath)
        old_result = f["result"]
        # TODO if any of result has same length as one of those in old_result, need to get their index 3 value and insert from new_result into old_result
        #for res in result
        #    for o_res in old_result
        #        if all(res[1] .== o_res[1]) && all(res[2] .== o_res[2])
        #            # TODO combine and pop out
        #            break
        #        end
        #    end
        #end
        new_result = vcat(old_result, result)
        result = sort(new_result, by=x->length(x[1]))
        JLD2.close(f)
        JLD2.jldsave(fpath; result)
    else
        println("new file")
        result = sort(result, by=x->length(x[1]))
        JLD2.jldsave(fpath; result)
    end
end


function parallel_sort!(result::Dict, by::String)
    ind, _ = zip(sort(enumerate(result[by]) |> collect, by=last)...) .|> collect
    for k in keys(result)
        if length(result[k]) == length(ind)
            result[k] = result[k][ind]
        end
    end
end


function save_probe_plot_result(fpath::String, result::Dict{String, Dict{String, Vector{Any}}})
    if isfile(fpath)
        println("Joining plot result")
        f = JLD2.jldopen(fpath, "a+")
        old_result = Dict(obj_name=>f[obj_name] for obj_name in keys(f))
        for (obj_name, obj_res) in result
            if obj_name in keys(old_result)
                old_res = old_result[obj_name]

                new_res = Dict()
                for k in keys(obj_res)
                    new_res[k] = vcat(old_res[k], obj_res[k])
                end
                parallel_sort!(new_res, "x")
                old_result[obj_name] = new_res
            else
                parallel_sort!(obj_res, "x")
                old_result[obj_name] = obj_res
            end
        end
        old_result = Dict(Symbol(k)=>v for (k, v) in old_result)
        JLD2.close(f)
        JLD2.jldsave(fpath; old_result...)
    else
        println("new file")
        for v in values(result)
            parallel_sort!(v, "x")
        end
        result = Dict(Symbol(k)=>v for (k, v) in result)
        JLD2.jldsave(fpath; result...)
    end
end


function pack_plot_variables(result::AbstractVector, obj_fn_names::Vector{String})
    plot_res = Dict(obj_name=>Dict(k=>[] for k in ["x", "train_bitcorr", "train_corr", "train_correct_samples", "bitcorr", "corr", "correct_samples"]) for obj_name in obj_fn_names)
    for (train, test, res) in result  # it's already sorted
        num_train = size(train)[1]
        for (obj_name, (net, optim_res, train_bitcorr, train_corr, train_correct_samples, bitcorr, corr, correct_samples)) in res
            push!(plot_res[obj_name]["x"], num_train)
            push!(plot_res[obj_name]["train_bitcorr"], train_bitcorr)
            push!(plot_res[obj_name]["train_corr"], train_corr)
            push!(plot_res[obj_name]["train_correct_samples"], train_correct_samples)
            push!(plot_res[obj_name]["bitcorr"], bitcorr)
            push!(plot_res[obj_name]["corr"], corr)
            push!(plot_res[obj_name]["correct_samples"], correct_samples)
        end
    end
    plot_res
end


function test_net(net::HopNet, test::Vector{G}) where {G<:BaseGraph}
    x = graph_data(test)
    out = convergedynamics(net, x)
    a = out .== x
    correct_samples = all(a, dims=2)
    sum(a) / length(a), sum(correct_samples) / length(correct_samples), correct_samples
end


# generate train and test datasets, do tests on all diff types of networks using the same train-test dataset pairs
function generate_iso_data(V::Int, K::Int, num_train::Int, num_test::Int, graph_generator::Function)
    g = graph_generator(V, K)

    train = sample_isomorphic(g, num_train)
    # ensure training samples number exactly num_train
    tries = 0
    while length(train) < num_train# && tries < 10
        new_train = sample_isomorphic(g, num_train)
        if length(new_train) > length(train)
           train = new_train
        end
        tries += 1
    end
    if length(train) < num_train
        println("Could not generate $(num_train) samples, going forward with $(length(train)) samples.")
    end
    test = sample_isomorphic(g, num_test)

    # Exhaustive test
    #train = all_isomorphic(g) |> Random.shuffle
    #test = all_isomorphic(g)

    train = train[1:min(num_train, length(train))]
    train, test
end


function generate_data(V::Int, K::Int, num_train::Int, num_test::Int, graph_generator::Function)
    # Random graph
    train = [graph_generator(V, K) for _ in 1:num_train]
    test = [graph_generator(V, K) for _ in 1:num_test]
    println("Generated $(length(train)) training samples independently.")
    train = train[1:min(num_train, length(train))]
    train, test
end


# for a given train-test dataset pair, train an OPR, delta, perceptron and full MEF network
function fit_networks(make_data_fn::Function, make_obj_fns)#::Dict{String, Tuple{Function, UnionAll}})
    results = Dict()
    train, test = make_data_fn()
    for (obj_name, (fit_obj_fn, optimiser)) in make_obj_fns
        net, result = fit_graphs(train, fit_obj_fn, optimiser)
        train_bitcorr, train_corr, train_correct_samples = test_net(net, train)
        bitcorr, corr, correct_samples = test_net(net, test)
        # store result and return
        results[obj_name] = net, result, train_bitcorr, train_corr, train_correct_samples, bitcorr, corr, correct_samples
        println("Done with $(obj_name) with $(size(train)) samples.")
    end
    train, test, results
end


# TODO make a continue_probe which gets original data, generates the biggest number of train_range and indexes the correct # samples that haven't been completed
# loop over range of # of train samples, do the above fitting
function probe_iso_train_range(V::Int, K::Int, train_range::StepRange, num_test::Int, graph_generator::Function, make_obj_fns)#::Dict{String, Tuple{Function, UnionAll}})
    train, test = generate_iso_data(V, K, train_range[end], num_test, graph_generator)
    train_over_range(train, test, train_range, make_obj_fns)
end


function probe_train_range(V::Int, K::Int, train_range::StepRange, num_test::Int, graph_generator::Function, make_obj_fns)
    train, test = generate_data(V, K, train_range[end], num_test, graph_generator)
    train_over_range(train, test, train_range, make_obj_fns)
end


function train_over_range(train, test, train_range, make_obj_fns)
    results = [[] for _ in 1:Threads.nthreads()]
    total_train_samples = length(train)
    # Adjust train_range to be the smallest range value in train_range bigger than total_train_samples
    train_range = train_range.start:train_range.step:min(filter(x->x>=total_train_samples, train_range)...)
    Threads.@threads for num_train in train_range
        make_data_fn = ()->(train[1:min(num_train, total_train_samples)], test)
        # store result and return
        result = fit_networks(make_data_fn, make_obj_fns)
        push!(results[Threads.threadid()], result)
        println("Done $num_train out of $(train_range[end])")
    end
    results |> Iterators.flatten |> collect
end


function probe_iso_train_range_proc(V::Int, K::Int, train_range::StepRange, num_test::Int, graph_generator::Function, make_obj_fns::Dict{String, Tuple{Function, UnionAll}})
    train, test = generate_iso_data(V, K, train_range[end], num_test, graph_generator)
    total_train_samples = length(train)
    make_data_fns = [()->(train[1:min(num_train, total_train_samples)], test) for num_train in train_range]
    obj_fns_dup = IterTools.ncycle([make_obj_fns], length(make_data_fns))
    Distributed.pmap(fit_networks, make_data_fns, obj_fns_dup)
end
