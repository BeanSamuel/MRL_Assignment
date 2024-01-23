import spicy
import numpy as np
data = spicy.io.loadmat('./rt_data.mat')
print(data.keys())
print(data['rt_data'].shape)
print(data['rt_data'][:3])