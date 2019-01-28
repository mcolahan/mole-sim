#%%
import numpy as np
import pandas as pd

radius = 0.5
x = np.array([9.75,2,3])
box_dims = np.array([[0,0,0], [10,10,10]])

dist = abs(box_dims - x)
(dist < radius).any()


#%%
x = (10,10,10)
dims = np.array([np.zeros(len(x)), x])
dims

#%%
