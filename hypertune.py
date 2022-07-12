#!/usr/bin/python
from PhIREGANs import *
import shutil

path_to_PhIRE = os.path.dirname(__file__)

def manyruns(runs, data_type, 
            mu_sig , r, data_path, model_path):
    '''
    Runs the phiregans model a specified number of times with default hyperparameter settings

    Saves all outputs and models in a subdir of structured_outs
    
    inputs(besides the standard phiregans inputs):
        runs: How many times to run the phiregans model
    outputs:
        ganarray: A numpy array of shape (runs,5,100,100,2)
    
    '''
    newdir = f'/hpcgpfs01/scratch/everdoore/structured_outs/mruns_{runs}_'+datetime.utcnow().strftime('%m%d-%H%M%S-%f')
    os.mkdir(newdir)

    new_data_dir = newdir+'/data_out'
    os.mkdir(new_data_dir)

    new_model_dir = newdir+'/models'
    os.mkdir(new_model_dir)
    i = 0
    while i < runs:
        phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig)

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
        
        data_out_path = path_to_PhIRE+'/data_out'
        mv_data = max([os.path.join(data_out_path,d) for d in os.listdir(data_out_path)], key=os.path.getmtime)
        shutil.move(mv_data, new_data_dir)

        model_save_path = path_to_PhIRE+'/models'
        mv_model = max([os.path.join(model_save_path,d) for d in os.listdir(model_save_path)], key=os.path.getmtime)
        shutil.move(mv_model, new_model_dir)
        i += 1
    
    data_list = [sorted(os.listdir(new_data_dir))[i]+'/dataSR.npy' 
        for i in range(len(sorted(os.listdir(new_data_dir))))]
    ganlist = [np.load(new_data_dir+'/'+data_list[i]) for i in range(len(data_list))]
    
    ganarray = np.asarray(ganlist)
    np.save(newdir+'/output_array',ganarray)

    return(ganarray)


def varyeps(epochlower, epochupper, data_type, 
            mu_sig , r, data_path, model_path):
    
    '''
        Runs the phiregans model iteratively with varying learning epoch number and returns
        an aggregate array of outputs of shape (k,5,100,100,2) where k = (epochupper - epochlower) +1

        Saves all outputs and models in a subdir of structured_outs
        
        inputs(besides the standard phiregans inputs):
            epochlower: Least number of learning epochs desired for the model to be run with
            epochupper: Greatest number of epochs desired for the model to be run with
        
        outputs:
            ganarray: A numpy array of shape (k,5,100,100,2)
    
    '''
    
    newdir = f'/hpcgpfs01/scratch/everdoore/structured_outs/mruns_{((epochupper-epochlower) + 1)}_'+datetime.utcnow().strftime('%m%d-%H%M%S-%f')
    os.mkdir(newdir)

    new_data_dir = newdir+'/data_out'
    os.mkdir(new_data_dir)

    new_model_dir = newdir+'/models'
    os.mkdir(new_model_dir)
    
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
        
        data_out_path = path_to_PhIRE+'/data_out'
        mv_data = max([os.path.join(data_out_path,d) for d in os.listdir(data_out_path)], key=os.path.getmtime)
        shutil.move(mv_data, new_data_dir)

        model_save_path = path_to_PhIRE+'/models'
        mv_model = max([os.path.join(model_save_path,d) for d in os.listdir(model_save_path)], key=os.path.getmtime)
        shutil.move(mv_model, new_model_dir)
    
    data_list = [sorted(os.listdir(new_data_dir))[i]+'/dataSR.npy' 
        for i in range(len(sorted(os.listdir(new_data_dir))))]
    ganlist = [np.load(new_data_dir+'/'+data_list[i]) for i in range(len(data_list))]
    
    ganarray = np.asarray(ganlist)
    np.save(newdir+'/output_array',ganarray)

    return(ganarray)

def varyrate(lower_rate, upper_rate, abs_lower_order_mag,
            data_type, mu_sig , r, data_path, model_path, 
            step = 'none'):
    '''
    Runs the phiregans model iterately with varying learning rates and returns an aggregate
    array of outputs of size (k,5,100,100,2) where k = number of models produced by the iteration
    
    Saves all outputs and models in a subdir of structured_outs

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

    if step == 'none':
        step = 1*(10**(-abs_lower_order_mag))
    vlist = np.arange(lower_rate, upper_rate+step, step)
    
    for i in range(len(vlist)):
        vlist[i] = round(vlist[i], abs_lower_order_mag)
    
    newdir = f'/hpcgpfs01/scratch/everdoore/structured_outs/{str(len(vlist))}vrate_{str(lower_rate)}-{str(step)}_'+datetime.utcnow().strftime('%m%d-%H%M%S-%f')
    os.mkdir(newdir)

    new_data_dir = newdir+'/data_out'
    os.mkdir(new_data_dir)

    new_model_dir = newdir+'/models'
    os.mkdir(new_model_dir)

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

        data_out_path = path_to_PhIRE+'/data_out'
        mv_data = max([os.path.join(data_out_path,d) for d in os.listdir(data_out_path)], key=os.path.getmtime)
        shutil.move(mv_data, new_data_dir)

        model_save_path = path_to_PhIRE+'/models'
        mv_model = max([os.path.join(model_save_path,d) for d in os.listdir(model_save_path)], key=os.path.getmtime)
        shutil.move(mv_model, new_model_dir)
    
    data_list = [sorted(os.listdir(new_data_dir))[i]+'/dataSR.npy' 
        for i in range(len(sorted(os.listdir(new_data_dir))))]
    ganlist = [np.load(new_data_dir+'/'+data_list[i]) for i in range(len(data_list))]
    
    ganarray = np.asarray(ganlist)
    np.save(newdir+'/output_array',ganarray)

    return(ganarray)