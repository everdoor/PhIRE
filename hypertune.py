from PhIREGANs import *

def varyeps(epochlower, epochupper, data_type, 
            mu_sig , r, data_path, model_path):
    '''
        Runs the phiregans model iteratively with varying learning epoch number and returns
        an aggregate array of outputs of shape (k,5,100,100,2) where k = (epochupper - epochlower) +1
        
        inputs(besides the standard phiregans inputs):
            epochlower: Least number of learning epochs desired for the model to be run with
            epochupper: Greatest number of epochs desired for the model to be run with
        
        outputs:
            ganarray: A numpy array of shape (k,5,100,100,2)
    
    '''
    
    dlist = []
    
    N = len(os.listdir('/home/emilio/bnl/climproj/PhIRE/data_out'))
    
    for i in range(epochlower,epochupper+1):
        phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig, N_epochs=i)
    
        model_dir = phiregans.pretrain(r=r,
                                   data_path=data_path,
                                   model_path=model_path,
                                   batch_size=1)

        model_dir = phiregans.train(r=r,
                                data_path=data_path,
                                model_path=model_dir,
                                batch_size=1)
    
        phiregans.test(r=r,
                   data_path=data_path,
                   model_path=model_dir,
                   batch_size=1, plot_data = True)
    
    K = len(os.listdir('/home/emilio/bnl/climproj/PhIRE/data_out'))
    
    for i in range(N,K):
        dlist.append(sorted(os.listdir('/home/emilio/bnl/climproj/PhIRE/data_out'))[i]+'/dataSR.npy')
    
    ganlist = []
    
    for i in range(len(dlist)):
        ganlist.append(np.load('/home/emilio/bnl/climproj/PhIRE/data_out/'+dlist[i]))
    
    ganarray = np.asarray(ganlist)
    return(ganarray)

def varyrate(lower_rate, upper_rate, abs_lower_order_mag,
            data_type, mu_sig , r, data_path, model_path, 
            step = 'none'):
    '''
    Runs the phiregans model iterately with varying learning rates and returns an aggregate
    array of outputs of size (k,5,100,100,2) where k = number of models produced by the iteration
    
    inputs(besides the standard phiregans inputs):
        lower_rate: The lower bound of learning rates which one wishes to run the model with
        upper_rate: The upper bound of learning rates which one wishes to run the model with
        lower_order_mag: This is the absolute value of the order of magnitude of the smaller learning rate,
            which is taken as input for rounding purposes. e.g. if one's lower bound for the learning rate is
            1e^-5, then abs_lower_order_mag = 5
        step: Since this code creates a list of learning rates with which to run the model, I've included
            an optional step with which to space the learning rates. If one does not enter a value for step
            it is set to 1e-(abs_lower_order_mag) by default
    outputs:

    '''
    
    dlist = []

    N = len(os.listdir('/home/emilio/bnl/climproj/PhIRE/data_out'))
    
    if step == 'none':
        step = 1*(10**(-abs_lower_order_mag))
    vlist = np.arange(lower_rate, upper_rate+step, step)
    
    for i in range(len(vlist)):
        vlist[i] = round(vlist[i], abs_lower_order_mag)

    for i in range(len(vlist)):
        phiregans = PhIREGANs(data_type= data_type, mu_sig=mu_sig, learning_rate= vlist[i])
    
        model_dir = phiregans.pretrain(r=r,
                                   data_path=data_path,
                                   model_path=model_path,
                                   batch_size=1)

        model_dir = phiregans.train(r=r,
                                data_path=data_path,
                                model_path=model_dir,
                                batch_size=1)
    
        phiregans.test(r=r,
                   data_path=data_path,
                   model_path=model_dir,
                   batch_size=1, plot_data = True)
                   
    K = len(os.listdir('/home/emilio/bnl/climproj/PhIRE/data_out'))
    
    for i in range(N,K):
        dlist.append(sorted(os.listdir('/home/emilio/bnl/climproj/PhIRE/data_out'))[i]+'/dataSR.npy')
    
    ganlist = []
    
    for i in range(len(dlist)):
        ganlist.append(np.load('/home/emilio/bnl/climproj/PhIRE/data_out/'+dlist[i]))
    
    ganarray = np.asarray(ganlist)
    return(ganarray)