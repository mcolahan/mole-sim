#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mayavi import mlab
# Create the data.
x = [1,2,3,4,5,6]
y = [1,2,3,4,5,6]
z = [1,2,3,4,5,6]

# View it.

s = mlab.points3d(x, y, z)
mlab.show()