from analysis import hyper_rmse
from hypertune import *



data_type = 'wind'
data_path = 'example_data/wind_LR-MR.tfrecord'
model_path = 'models/wind_lr-mr/trained_gan/gan'
r = [2, 5]
mu_sig=[[0.7684, -0.4575], [4.9491, 5.8441]]

y = np.load('/home/emilio/bnl/climproj/PhIRE/true_data/wind_LR-MR.npy')

x = varyeps(5,10, data_type=data_type, mu_sig=mu_sig, r = r, data_path= data_path, model_path= model_path)

z = hyper_rmse(x, y)

for i in range(6):
    print(f'\nThis is the RMSE array for the model with {i+5} training epochs:\n')
    print(z[i,:,:])


