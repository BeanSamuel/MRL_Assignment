import spicy

data = spicy.io.loadmat('./Labeled_data/Labeled_data/rt_data.mat')
print(data['rt_data'].shape)
print(data['rt_data'][0:2])

