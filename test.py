#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%

box_size = np.array([8,8,8])
n_parts = 512
n_dims = 3
min_dist = 0.8

init_locs = np.random.rand(n_parts, n_dims) * box_size
min_dists = []
for i in range(n_parts):
    continue_looping = True
    if i == 0:
        continue
    loop_count = 0
    while continue_looping == True:
        if loop_count > 1000:
            raise InterruptedError('Loop was interrupted')

        dr = init_locs[i, :] - init_locs[:i, :]
        dr = dr - np.around(dr / box_size) * box_size  # PBC
        dist = np.sqrt(np.sum(dr ** 2, axis=1))

        if (dist < min_dist).any() == True:
            init_locs[i, :] = np.random.rand(1, n_dims) * box_size
        else:
            continue_looping = False
            min_dists.append(min(dist))

        loop_count += 1

print(min(min_dists))

r = init_locs

count = 0
for i in range(n_parts-1):
    # get position vectors and distances between particles
    r_ij_vect = r[i, :] - r[(i + 1):, :]
    r_ij_vect -= np.around(r_ij_vect / box_size) * box_size  # pbc
    r_ij = np.sqrt(np.sum(r_ij_vect ** 2, axis=1))  # magnitude (norm) of r_ij_vect
    
    count += np.sum(r_ij < min_dist)
    
print(count)
    


#%%
box_size = np.array([4,4,4])
x = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
dx = x[0,:] - x[(i+1):,:]


#%%
r_ij = np.array([3,4,5,6,7,8,9])
np.sum(r_ij < 6)

#%%

