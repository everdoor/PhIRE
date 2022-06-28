#!/usr/bin/python
from PhIREGANs import *

tf.enable_eager_execution()

def extract_fn(data_record):
    features = {
        # Extract features using the keys set during creation
        'index': tf.FixedLenFeature([], tf.int64),
        'data_LR': tf.FixedLenFeature([],tf.string),
        'h_LR': tf.FixedLenFeature([], tf.int64),
        'w_LR': tf.FixedLenFeature([], tf.int64),
        'data_HR': tf.FixedLenFeature([], tf.string),
        'h_HR': tf.FixedLenFeature([], tf.int64),
        'w_HR': tf.FixedLenFeature([], tf.int64),
        'c': tf.FixedLenFeature([], tf.int64)
        }
    content = tf.parse_single_example(data_record, features)

    index = content['index']
    data_LR = content['data_LR']
    h_LR = content['h_LR']
    w_LR = content['w_LR']
    data_HR = tf.decode_raw(content['data_HR'], tf.float64)
    h_HR = content['h_HR']
    w_HR = content['w_HR']
    c = content['c']

    data_HR = tf.reshape(data_HR, (h_HR, w_HR, c))

    return(data_HR)

def get_ds(file_name):
    ds = tf.data.TFRecordDataset(file_name)
    ds = ds.map(
        extract_fn
    )
    return(ds)

dataset = get_ds('/home/emilio/bnl/climproj/PhIRE/example_data/solar_LR-MR.tfrecord')

trudata = []
for i in dataset:
    a = i.numpy()
    trudata.append(a)
trudata = np.asarray(trudata)
np.save('/home/emilio/bnl/climproj/PhIRE/true_data/solar_LR-MR',trudata)









