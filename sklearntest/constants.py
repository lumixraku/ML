import numpy as np

my8 = [0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 10, 10, 1, 0, 0,
       0, 0, 10, 0, 0, 10, 0, 0,
       0, 0, 3, 10, 10, 3, 0, 0,
       0, 0, 3, 10, 10, 3, 0, 0,
       0, 0, 10, 0, 0, 10, 0, 0,
       0, 0, 1, 10, 10, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0]
my82D = np.asarray(my8).reshape(1, 64)