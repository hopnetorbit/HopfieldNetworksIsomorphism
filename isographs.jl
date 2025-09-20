import Pkg
Pkg.activate(".")
import Combinatorics: combinations, permutations, Permutations
import Random: shuffle
import Memoize: @memoize
import LRUCache: LRU
import JLD2: jldsave


abstract type BaseGraph end


struct GraphBits<:BaseGraph
    V::Int
    bitstring::Vector{UInt8}

    function GraphBits(V::Int, chosen_edge_i::Vector{Int})
        g = Graph(V, chosen_edge_i)
        new(V, g.bitstring)
    end
end


struct Graph<:BaseGraph
	V::Int
    bitstring::Vector{UInt8}
    edges::Vector{Vector{Int}}

    function Graph(V, chosen_edge_i, all_edges=nothing)
        all_edges = isnothing(all_edges) ? make_edges(V) : all_edges
        bitstring = zeros(UInt8, length(all_edges))
        bitstring[chosen_edge_i] .= 1
        chosen_edges = all_edges[chosen_edge_i]
        new(V, bitstring, chosen_edges)
    end

    function Graph(V::Int, chosen_edges::Vector{Vector{Int}})
        chosen_edges = unique(sort.(chosen_edges))
        all_edges = make_edges(V)
        edge_i = Dict(K=>i for (i, K) in enumerate(all_edges))
        chosen_edge_i = [edge_i[e] for e in chosen_edges]
        Graph(V, chosen_edge_i)
    end
end


"""Concatenates all bitstrings of each graph in graphs into a BitMatrix"""
graph_data(graphs::Vector{BG}) where {BG<:BaseGraph} = hcat([i.bitstring for i in graphs]...)' |> collect


Base.:(==)(g1::GT, g2::GT) where {GT<:BaseGraph} = g1.V == g2.V && all(g1.bitstring .== g2.bitstring)
Base.isequal(g1::GT, g2::GT) where {GT<:BaseGraph} = g1 == g2
Base.hash(g::GT) where {GT<:BaseGraph} = hash(g.bitstring)


"""Gets every 2-combination of nodes in 1:V"""
# We want to cache this because we'll probably be calling this function many times while creating graphs of the same size
@memoize LRU{Tuple{Int}, Vector{Vector{Int}}}(maxsize=2) make_edges(V::Int)::Vector{Vector{Int}} = combinations(1:V, 2) |> collect


"""Samples n objects from collection A without replacement"""
function sample_wo_repl!(A, n::Int)
    sample = eltype(A)[]
    for i in 1:n
        push!(sample, splice!(A, rand(eachindex(A))))
    end
    return sample
end


function random_group_graph(V::Int, K::Int)::Graph
    num_edges = V * (V-1) / 2 |> Int
    e = num_edges / 2 |> Int
    random_graph(V, e)
end


"""
V is the number of nodes in the graph, e is the number of edges to connect.
"""
function random_graph(V::Int, e::Int)::Graph
    num_edges = V * (V-1) / 2 |> Int
    @assert(num_edges >= e, "$(e) is too many edges for graph of size $(V).")
    chosen_edge_i = sample_wo_repl!(1:num_edges |> collect, e) |> sort
    Graph(V, chosen_edge_i)
end


"""
V is the number of nodes in the graph, K is the number of nodes in one partition.
"""
function random_bipartite(V::Int, K::Int)
    @assert(K < V, "Only graph of size $(V) but $(K) nodes requested in one partition.")
    nodes = shuffle(1:V)
    part1 = nodes[1:K]
    part2 = nodes[K+1:end]
    chosen_edges = [[i, j] for i in part1 for j in part2]
    Graph(V, chosen_edges)
end


"""
V is the number of nodes in the graph, K is the number of nodes in the clique.
"""
function random_clique(V::Int, K::Int)
    @assert(K <= V, "Only graph of size $(V) but $(K) nodes requested in the clique.")
    nodes = shuffle(1:V)
    clique_node_inds = nodes[1:K]
    chosen_edges = [[i, j] for i in clique_node_inds for j in clique_node_inds if i != j]
    Graph(V, chosen_edges)
end


"""
V is the number of nodes in the graph. It's actually also not random.
"""
function paley_graph(V::Int, K::Int=0)::Graph # we don't care about K
    square_set = Set([(x^2)%V for x in 1:V-1 if (x^2)%V != 0])
    chosen_edges = Vector{Int}[]
    for x in 0:V-1
        for x2 in square_set
            push!(chosen_edges, [x+1, (x+x2)%V+1])
        end
    end
    Graph(V, chosen_edges)
end


"""
Produces a map of edge (vertex pairs) to index in bitstring
"""
function edge_indices(V::Int)::Dict{Vector{Int}, Int}
    all_edges = make_edges(V)
    Dict(e=>i for (i, e) in enumerate(all_edges))
end


"""
Extract sorted unique nodes that occur in g.edges
"""
function get_attached_nodes(g::Graph)::Vector{Int}
    vcat(g.edges...) |> unique |> sort
end


"""
Returns a function that shuffles a set of indices of nodes num_samples times
"""
function shuffle_n(num_samples::Int)
    shuff(nodes::Vector{Int}) = [shuffle(nodes) for _ in 1:num_samples]
end


"""
Shuffle only nodes that occur in g.edges using shuffle_fn which returns a set of shuffled indices, reinsert based on original index
"""
# TODO this is buggy, or at least doesn't do what we need it to
function shuffle_attached_nodes(g::Graph, shuffle_fn::Function)::Vector{Vector{Int}}
    attached_nodes = get_attached_nodes(g)
    all_nodes = 1:g.V |> collect
    shuffled_nodes = shuffle_fn(attached_nodes)
    permed_edges = Vector{Int}[]
    for shuf in shuffled_nodes
        new_perm = copy(all_nodes)
        for (i, j) in zip(attached_nodes, shuf)
            new_perm[i] = j
        end
        push!(permed_edges, new_perm)
    end
    permed_edges
end


"""
Returns all Graph isomorphic to g
"""
function all_isomorphic(g::G, T::DataType=Graph)::Vector{T} where {G<:BaseGraph}
    #permed_nodes = shuffle_attached_nodes(g, permutations)
    permed_nodes = permutations(1:g.V)
    sample_isomorphic(g, permed_nodes, T) |> unique
end


"""
Randomly sample num_samples GraphBits that are isomorphic to g
"""
function sample_isomorphic(g::Graph, num_samples::Int)::Vector{GraphBits}
    #shuffle_fn = shuffle_n(num_samples)
    #permed_nodes = shuffle_attached_nodes(g, shuffle_fn)
    permed_nodes = shuffle_n(num_samples)(1:g.V |> collect)
    sample_isomorphic(g, permed_nodes)
end


"""
Given a graph g and a set of vectors of permuted nodes, return the isomorphic GraphBits corresponding to permuted nodes
"""
function sample_isomorphic(g::Graph, permed_nodes::Union{Vector{Vector{Int}}, Permutations{UnitRange{Int}}}, T::DataType=GraphBits)::Vector{T}
    @assert(all(g.V .== length.(permed_nodes)))
    chosen_edges = g.edges
    edge_to_index = edge_indices(g.V)
    chosen_permed_edges = Set([[[pn[ni] for ni in e] |> sort for e in chosen_edges] |> sort for pn in permed_nodes]) |> collect
    [T(g.V, [edge_to_index[e] for e in cpe]) for cpe in chosen_permed_edges]
end


if abspath(PROGRAM_FILE) == @__FILE__
    V = 4
    all_edges = make_edges(V)
    chosen_edges = [[1,3], [1,4], [2, 3]]
    #edge_i = Dict(K=>i for (i, K) in enumerate(all_edges))
    #chosen_edge_i = [edge_i[e] for e in chosen_edges]
    println(chosen_edges)
    a = Graph(V, chosen_edges)
    b = all_isomorphic(a)
    println(b)


    c = paley_graph(9) # 10 takes a long time to generate isomorphic graphs
    println(c)
    r = all_isomorphic(c)
    println(length(r))

    d = random_clique(10, 5)
    e = random_bipartite(10, 5)
end

