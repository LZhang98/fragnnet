# frag-gnn
Mass spectrometry prediction using combinatorial fragmentation and graph neural networks.


## Used libs
### SpectralEntropy
Publication [https://doi.org/10.1038/s41592-021-01331-z]
Github ['https://github.com/YuanyueLi/SpectralEntropy']
Doc [https://spectralentropy.readthedocs.io/en/master/]

## For developers
### run memory check
valgrind --tool=memcheck --leak-check=full --suppressions=./data/valgrind-python.supp  --log-file=minimal.valgrind.log python ./sc
ripts/run_compute_frags.py 
### check memory usage
memray run --native ./scripts/run_compute_frags.py
memray flamegraph ./scripts/run_compute_frags.abc.bin

### Profile
[https://medium.com/@narenandu/profiling-and-visualization-tools-in-python-89a46f578989]
~ pip install snakeviz
snakeviz run_frag_exp.prof