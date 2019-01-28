#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#%%
n_parts = 10
n_dims = 2
box_size = (10,10)
box_lims = np.array([np.zeros(len(box_size)), box_size])

time = (0, 50)
n_times = 5

part_locs_init = np.random.rand(n_parts, n_dims) * box_size
part_vel = np.array([np.ones(n_parts), np.zeros(n_parts)]).T
times = np.linspace(time[0], time[1], n_times)

part_locs = np.zeros((n_parts, n_dims, n_times))

for n in range(n_times):
    if n == 0:
        part_locs[:, :, n] = part_locs_init
        continue

    delta_t = times[n] - times[n - 1]
    new_part_loc = part_locs[:, :, n-1] + part_vel * delta_t
    for i in range(n_dims):
        new_part_loc[new_part_loc[:,i] > box_size[i], i] -= box_size[i]
        new_part_loc[new_part_loc[:,i] < 0, i] += box_size[i]
    
    part_locs[:, :, n] = new_part_loc



fig, ax = plt.subplots()
x, y = [], []
pts, = plt.plot([],[], 'o', animated=True)

def init():
    ax.set_xlim(0, box_size[0])
    ax.set_ylim(0, box_size[1])
    return pts,

def update(frame):
    x = part_locs[:, 0, frame]
    y = part_locs[:, 1, frame]
    pts.set_data(x, y)
    return pts,

ani = FuncAnimation(fig, update, frames=np.arange(n_times), 
    init_func=init, blit=True)
plt.show()
# part_locs

# #%%
# box_size = (10,10,10)
# box_lims = np.array([np.zeros(len(box_size)), box_size])

# new_part_loc = np.array([[1,-1,3],[4,5,6],[7,8,9], [10,11,12]])
# for n in range(n_dims):
#     new_part_loc[new_part_loc[:,n] > box_size[n], n] -= box_size[n]
#     new_part_loc[new_part_loc[:,n] < 0, n] += box_size[n]
# new_part_loc

#%%
