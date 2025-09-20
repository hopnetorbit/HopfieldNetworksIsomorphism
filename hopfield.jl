include("isographs.jl")
import Optim
import CUDA: cu
import LinearAlgebra
import Zygote: withgradient
import NPZ


struct HopNet{F<:AbstractFloat}
    n::Int
    V::Int
    W1::AbstractVector{F}
    W2::AbstractMatrix{F}

    function HopNet(V::Int, params::Vector{F}) where {F<:AbstractFloat}
        n = V * (V-1) / 2 |> Int
        W1 = params[1:n]
        W2 = reshape(params[n+1:end], n, n)
        new{F}(n, V, W1, W2)
    end


    function HopNet(V::Int)
        params = zeros(n + n ^ 2)
        HopNet(V, params)
    end
end


function convergedynamics(net::HopNet, x::AbstractMatrix; iterate_nodes=nothing, max_iter=100)
    iterate_nodes = isnothing(iterate_nodes) ? (1:size(x)[2]) : iterate_nodes
    convergedynamics(net.W1, net.W2, x, iterate_nodes, max_iter)
end


function convergedynamics(W1::AbstractVector{F}, W2::AbstractMatrix{F}, x::AbstractMatrix, iterate_nodes, max_iter) where {F<:AbstractFloat}
    out_ = dynamics(W1, W2, x, iterate_nodes)
    out = dynamics(W1, W2, out_, iterate_nodes)
    cnt = 0
    while !all(out .== out_) && cnt < max_iter
        out_ .= out
        out = dynamics(W1, W2, out_, iterate_nodes)
        cnt += 1
    end
    out
end


function dynamics(W1::AbstractVector{F}, W2::AbstractMatrix{F}, x::AbstractMatrix, iterate_nodes) where {F<:AbstractFloat}
    x_ = x[:, :]
    for i in iterate_nodes
        out = dyn_step(W1[i], W2[i, :], x_)
        x_[:, i] .= out .> 0
    end
    x_
end


function dyn_step(W1::F, W2::AbstractVector{F}, x_) where {F<:AbstractFloat}
    x_ * W2 .+ W1
end


function mef_obj(W1::AbstractVector{F}, W2::AbstractMatrix{F}, x, flipcounts, γ) where {F<:AbstractFloat}
    out = flipcounts .* ((x * W2)' .+ W1)'

    exp.(clamp.(out, -10, 10))
end


function fit_graphs(graphs::Vector{G}, make_obj_fn::Function, optimiser=Optim.LBFGS) where {G<:BaseGraph}
    g1 = first(graphs)
    x = graph_data(graphs) .|> Int8
    fit_2nd_order(g1.V, x, make_obj_fn, optimiser)
end


function fit_2nd_order(V::Int, x::Matrix, make_obj_fn::Function, optimiser=Optim.LBFGS)
    m, n = size(x)
    #obj = make_obj_fn(n, x, ones(m))
    W = zeros(n + n^2)
    #result = Optim.optimize(obj, W, optimiser())
    #Ŵ = Optim.minimizer(result)
    Ŵ, result = make_obj_fn(W, x, ones(m), optimiser)
    net = HopNet(V, Ŵ)
    net, result
end


function make_fg(obj_fn)
    function fg(F, G, W)
        l, g = withgradient(obj_fn, W)
        if G !== nothing
            G .= g[1]
        end
        if F !== nothing
            return l
        end
    end

    Optim.only_fg!(fg)
end


function fit_mef_2nd_order(W, x, counts, optimiser; iterate_nodes=nothing, array_cast=Array)
    m, n = size(x)
    obj = make_2nd_order_mef_obj(n, x, counts; iterate_nodes, array_cast=array_cast)
    result = Optim.optimize(obj, W, optimiser())
    Ŵ = Optim.minimizer(result)
    Ŵ, result
end


function make_2nd_order_mef_obj(n, x, counts; iterate_nodes=nothing, array_cast=Array)
    iterate_nodes = isnothing(iterate_nodes) ? (1:n) : iterate_nodes
    flip = 1 .- 2 .* x
    flipcounts = flip .* counts
    mask = 1 .- LinearAlgebra.I(n)
    function obj(W)
        W1 = W[1:n]
        W2 = reshape(W[n+1:end], n, n)
        W2 = (W2 + W2') ./ 2 .* mask
        out = sum(mef_obj(W1, W2, x, flipcounts, 0))
        out
    end

    make_fg(obj)
end


function fit_delta(W, x, counts, optimiser; iterate_nodes=nothing, array_cast=Array)
    m, n = size(x)
    obj = make_delta_obj(n, x, counts; iterate_nodes=iterate_nodes, array_cast=array_cast)
    result = Optim.optimize(obj, W, optimiser())
    Ŵ = Optim.minimizer(result)
    Ŵ, result
end


sigmoid(z) = 1 ./ (1 .+ exp.(-z))


function make_delta_obj(n, x, counts; iterate_nodes=nothing, array_cast=Array)
    iterate_nodes = isnothing(iterate_nodes) ? (1:n) : iterate_nodes
    m, n = size(x)
    mask = 1 .- LinearAlgebra.I(n)
    function delta_obj(W)
        W1 = reshape(W[1:n], (1, n))
        W2 = reshape(W[n+1:end], (n, n)) .* mask
        #h = x * W2 .+ W1
        h = (W2 * x')' .+ W1
        ŷ = sigmoid(h)
        sum((x - ŷ).^2) / 2
    end

    make_fg(delta_obj)
end


function fit_perceptron(W, x, counts, optimiser; iterate_nodes=nothing, array_cast=Array)
    m, n = size(x)
    W1 = W[1:n]
    W2 = reshape(W[n+1:end], n, n)
    mask = 1 .- LinearAlgebra.I(n)
    num_epoch = 1000
    lr = 0.001
    time_run = 0.
    for _ in 1:num_epoch
        for xb in eachrow(x)
            start = time()
            perceptron_step!(W1, W2, xb, mask, lr)
            time_run += time() - start
        end
    end
    Ŵ = vcat(W1, vec(W2'))
    Ŵ, (; time_run)
end


function perceptron_step!(W1, W2, x, mask, lr)
    h = W2 * x + W1
    ŷ = h .> 0
    diff = x - ŷ
    W1 .+= lr .* diff
    W2 .+= lr .* (diff * x') .* mask
end


function make_perceptron_obj(n, x, counts; iterate_nodes=nothing)
    iterate_nodes = isnothing(iterate_nodes) ? (1:n) : iterate_nodes
    m, n = size(x)
    mask = 1 .- LinearAlgebra.I(n)
    function obj_val(W)
        W1 = reshape(W[1:n], (1, n))
        W2 = reshape(W[n+1:end], (n, n))
        h = (W2 * x')' .+ W1
        ŷ = h .> 0
        x - ŷ
    end

    function perceptron_obj(W)
        sum(obj_val(W))
    end

    function fg(F, G, W)
        if G !== nothing
            diff = obj_val(W)
            G[1:n] .= vec(sum(diff, dims=1))
            g2 = sum(d1 * x1' for (x1, d1) in zip(eachrow(x), eachrow(diff))) .* mask
            G[n+1:end] .= vec(g2)
        end
        if F !== nothing
            return perceptron_obj(W)
        end
    end
    Optim.only_fg!(fg)
end


if abspath(PROGRAM_FILE) == @__FILE__
    array_cast = Array  # apparently CPU is faster than GPU with Optim since we need to enable scalar operations

    #x = [[1 0 1 0]; [0 1 0 1]] |> array_cast
    #m, n = size(x)
    #W = zeros(n + n ^ 2) |> array_cast
    #counts = ones(m) |> array_cast
    #to_opt = make_2nd_order_mef_obj(n, x, counts)
    #results = Optim.optimize(to_opt, W, Optim.AcceleratedGradientDescent())
    #Ŵ = Optim.minimizer(results)
    #println(Ŵ)
    #out = convergedynamics(Ŵ, x |> Array)
    #println("$(all(out .== x))")

    # test out graphs of diff sizees
    K = 14
    V = 8
    s = [random_graph(V, K) for _ in 1:35]
    #s = sample_isomorphic(s[1], 10)
    println(length(s))
    x = graph_data(s) .|> Int8
    #x = NPZ.npzread("sample_random.npy")[1:7, :]
    times = [[] for _ in 1:Threads.nthreads()]
    Threads.@threads for i in 1:2*Threads.nthreads()
        # GradientDescent works fine for delta, whether using row or column of W2 doesn't matter to either delta nor mef
        #net, results = fit_graphs(s, fit_mef_2nd_order, ()->Optim.Adam(alpha=0.001))
        net, results = fit_graphs(s, fit_perceptron, Optim.GradientDescent)

        #local m, n = size(x)
        #W, results = fit_perceptron(zeros(n+n^2), x, ones(m), Optim.GradientDescent)
        #net = HopNet(V, W)

        push!(times[Threads.threadid()], results.time_run)

        #print(results)
        #Ŵ = Optim.minimizer(results)
        #println(sum(abs.(Ŵ)))

        #W = NPZ.npzread("wtf_W.npy")
        #n = 28
        #W = vcat(W[1:n], vec(reshape(W[n+1:end], n, n)'))
        #net = HopNet(V, W)
        x̂ = convergedynamics(net, x)
        res = x .== x̂
        println("$(sum(res)) $(length(res)) $(sum(all(res, dims=2)))")
    end
    times = times |> Iterators.flatten |> collect
    println(times)
    println(sum(times) / length(times))
end
