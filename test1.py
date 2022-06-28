from hypertune import varyrate
from analysis import *

data_type = 'wind'
data_path = 'example_data/wind_LR-MR.tfrecord'
model_path = 'models/wind_lr-mr/trained_gan/gan'
r = [2, 5]
mu_sig=[[0.7684, -0.4575], [4.9491, 5.8441]]

x = np.load('/home/emilio/bnl/climproj/PhIRE/true_data/wind_LR-MR.npy')
y = varyrate(.00008, .00012, 5, data_type = data_type, mu_sig=mu_sig, r = r,
 data_path = data_path, model_path=model_path, step = .00001)
z = hyper_rmse(y, x)

for i in range(4):
    print(f'\nThis is the RMSE array for the model with learning rate = {.00008 + i*.00002}:\n')
    print(z[i,:,:])