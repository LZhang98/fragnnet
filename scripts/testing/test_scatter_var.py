import torch as th

from frag_gnn.utils.misc_utils import scatter_var

x = th.tensor(
    [
        1., 1., 1.,
        1., 2., 3.,
        1.,
        0., 3., 0., 3.
    ]
)

b = th.tensor(
    [
        0, 0, 0,
        1, 1, 1,
        2,
        3, 3, 3, 3
    ]
)

v = scatter_var(x, b)

# 0., 1., 0., 3.
print(v)