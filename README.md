# frag-gnn
Mass spectrometry prediction using combinatorial fragmentation and graph neural networks.

## For developers
### run memory check
valgrind --tool=memcheck --leak-check=full --suppressions=./data/valgrind-python.supp  --log-file=minimal.valgrind.log python ./sc
ripts/run_compute_frags.py 
### check memory usage
memray run --native  python ./scripts/run_compute_frags.py