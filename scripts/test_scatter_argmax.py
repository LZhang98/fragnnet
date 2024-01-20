import torch as th

from frag_gnn.utils.misc_utils import scatter_argmax

x = th.tensor(
    [
        0, 1, 2, 3,
        2, 1, 0,
        1, 1, 1,
        0, 1,
        1, 0,
        1, 0, 1
    ]
)

b = th.tensor(
    [
        0, 0, 0, 0,
        1, 1, 1,
        2, 2, 2,
        3, 3,
        4, 4,
        5, 5, 5
    ]
)

i = th.tensor(
    [
        0, 1, 2, 3,
        0, 1, 2,
        0, 1, 2,
        0, 1,
        0, 1,
        0, 1, 2
    ]
)

am = scatter_argmax(x, b, i, dim_size=7)

# 3, 0, 2, 1, 0, 2
print(am)