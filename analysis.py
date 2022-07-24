#!/usr/bin/python
from math import log10
import numpy as np

def rmse(ground_truth_array, gan_array):
    '''
        Compute RMSE of image data
        inputs:
            ground_truth_array: array of inputs for the model pulled from tfrecord of shape (5,100,100,2)
            gan_array: array of GAN outputs from the model of shape (5,100,100,2)
        outputs:
            rmse: a matrix of size (5,2) whose (i,j)th entry is the RMSE of the images in position (i,:,:,j)
    '''
    
    trudata = ground_truth_array
    gandata = gan_array
    
    n_rmse = []
    e_rmse = []

    for j in range(2):
        for i in range(5):
            diff = trudata[i,:,:,j] - gandata[i,:,:,j]
            diffsq = diff*diff
            rmse_num = np.sum(diffsq)
            true_mean = np.mean(trudata[i,:,:,j])
            true_mean_array = np.full((100,100), true_mean)
            dif = gandata[i,:,:,j] - true_mean_array
            sq = dif*dif
            rmse_denom = np.sum(sq)
            rmse = rmse_num/rmse_denom
            if j == 0:
                n_rmse.append(rmse)
            if j == 1:
                e_rmse.append(rmse)
    rmse = [n_rmse, e_rmse]
    rmse = np.transpose((np.asarray(rmse)))
    return(rmse)

def hyper_rmse(ground_truth_array,hyper_ganarray):
    '''
    Computes RMSE for the outputs of multiple models that have been aggregated into an array

    inputs:
        hyper_ganarray: An array of shape (k,5,100,100,2). This array is to be interpretted
        as an array of outputs from multiple models that were produced by one of the functions in
        hypertune.py, the index k corresponds to the number of models.
    outputs: An array of shape (k,5,2) whose (n,i,j)th entry corresponds to the RMSE of the images
        produced by the n'th model in positions [i,:,:,j] of the relevant arrays.
    
    '''
    rmse_array_list = [rmse(ground_truth_array, hyper_ganarray[k,:,:,:,:]) for k in range(0,np.shape(hyper_ganarray)[0])]
    hyper_rmse_array = np.asarray(rmse_array_list)
    return(hyper_rmse_array)

def psnr(ground_truth_array,gan_array):
    
    gandata=gan_array
    trudata=ground_truth_array

    n_psnr=[]
    e_psnr=[]

    for j in range(2):
        for i in range(5):
            max_value=np.max(gandata[i,:,:,j])
            diff = trudata[i,:,:,j] - gandata[i,:,:,j]
            diffsq = diff*diff
            mse = np.sum(diffsq)/1000
            psnr = 20*log10(max_value) -10*log10(mse)
            if j==0: n_psnr.append(psnr)
            if j==1: e_psnr.append(psnr)
    psnr=[n_psnr,e_psnr]
    psnr = np.transpose(np.asarray(psnr))
    return(psnr)

def hyper_psnr(ground_truth_array,hyper_ganarray):
 
    psnr_array_list = [psnr(ground_truth_array, hyper_ganarray[k,:,:,:,:]) for k in range(0,np.shape(hyper_ganarray)[0])]
    hyper_psnr_array = np.asarray(psnr_array_list)
    return(hyper_psnr_array)