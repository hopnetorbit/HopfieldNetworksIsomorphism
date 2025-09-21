# set up output and plot directories
subrun="test"
mkdir -p results/$subrun/{random_clique,random_bipartite,paley_graph,random_group_graph}
mkdir plots

# install packages in Project.toml
julia -e "import Pkg; Pkg.activate(\".\"); Pkg.instantiate()"

# comparison experiments
uv run batchprobe.py $subrun MEF,Delta,Perceptron paley_graph 8 10 10 100 1 10
uv run batchprobe.py $subrun MEF,Delta,Perceptron random_bipartite 8 5 5 35 1 10
uv run batchprobe.py $subrun MEF,Delta,Perceptron random_clique 8 5 5 70 1 10

uv run batchprobe.py $subrun MEF,Delta,Perceptron paley_graph,random_bipartite,random_group_graph 20 200 100 700 1 10
uv run batchprobe.py $subrun MEF,Delta,Perceptron random_clique 20 100 100 600 1 10

# scaling experiments
uv run batchprobe.py $subrun MEF paley_graph 15 100 5 200 1 10
uv run batchprobe.py $subrun MEF paley_graph 20 250 5 350 1 10
uv run batchprobe.py $subrun MEF paley_graph 25 450 5 550 1 10
uv run batchprobe.py $subrun MEF paley_graph 30 700 5 770 1 10
uv run batchprobe.py $subrun MEF paley_graph 35 1000 10 1100 1 10
uv run batchprobe.py $subrun MEF paley_graph 30 1400 10 1600 1 10

uv run batchprobe.py $subrun MEF random_bipartite 15 200 5 250 1 10
uv run batchprobe.py $subrun MEF random_bipartite 20 275 10 345 1 10
uv run batchprobe.py $subrun MEF random_bipartite 25 600 25 800 1 10
uv run batchprobe.py $subrun MEF random_bipartite 30 700 10 760 1 10
uv run batchprobe.py $subrun MEF random_bipartite 35 1600 100 2600 1 10
uv run batchprobe.py $subrun MEF random_bipartite 30 1300 50 1600 1 10

uv run batchprobe.py $subrun MEF random_clique 15 5 5 95 1 10
uv run batchprobe.py $subrun MEF random_clique 20 50 10 150 1 10
uv run batchprobe.py $subrun MEF random_clique 25 70 10 180 1 10
uv run batchprobe.py $subrun MEF random_clique 30 90 10 200 1 10
uv run batchprobe.py $subrun MEF random_clique 35 110 10 220 1 10
uv run batchprobe.py $subrun MEF random_clique 30 110 10 300 1 10
