from IPython.display import display, Markdown

import sys
sys.path.insert(0, '../py_scripts')

import time
import numpy as np
import scipy as sp
import pandas as pd
import scipy.optimize as opt
import numpy.random as rand

import matplotlib.pyplot as plt
import seaborn as sns

# import push_pull as pp
import noise_models as noise
import thermo_models as thermo



def setup_model_params(df_dataset_key, df_params=None, noise_models=None):
    

    # map of prefit param names to values
    prefit_params = {}
    if df_params is not None:
        for index, row in df_params.iterrows():
            prefit_params[row['name']] = row['val']
    
    

    # map of each dataset to list of params 
    dataset_to_params = {}
    # map of parameters names to indices
    param_to_index = {}
    
    x0 = []
    bounds = []
    
    if noise_models is not None:
        
        dataset_to_params['noise_model'] = []
        
        if isinstance(noise_models['phospho']['GFP2MOCU'], noise.LogNormalBGNoise):
            # first set loss function parameters

            if 'phospho_sigma' not in prefit_params and 'phospho_sigma' not in param_to_index:
                param_to_index['phospho_sigma'] = len(x0)
                x0.append(2.0)
                bounds.append((0.5, 2.0))

            dataset_to_params['noise_model'].append('phospho_sigma')
            
        
        for exp_name, row in df_dataset_key.iterrows():
            model = row['model']
            if model == 'two_layer' or model == 'two_layer_nowriter' or model == 'two_layer_noeraser':  
                if isinstance(noise_models['kinase2_phospho']['GFP2MOCU'], noise.LogNormalBGNoise):
                    # first set loss function parameters

                    if 'kinase2_phospho_sigma' not in prefit_params and 'kinase2_phospho_sigma' not in param_to_index:
                        param_to_index['kinase2_phospho_sigma'] = len(x0)
                        x0.append(2.0)
                        bounds.append((0.5, 2.0))

                    dataset_to_params['noise_model'].append('kinase2_phospho_sigma')
                    
                    break

    
    for exp_name, row in df_dataset_key.iterrows():
    
    
        dataset_to_params[exp_name] = []
    
        model = row['model']
        
        # all models have a background noise model parameters

        
        # background phospho/dephospho rates
            
        if model == 'substrate_only' or model == 'push' or model == 'pushpull':
            
            # substrate bg phospho rate
            if 'sub_bg_phospho_rate' not in prefit_params and 'sub_bg_phospho_rate' not in param_to_index:
                param_to_index['sub_bg_phospho_rate'] = len(x0)
                x0.append(-2.0)
                bounds.append((-8, 1))

            dataset_to_params[exp_name].append('sub_bg_phospho_rate')
            
        if model == 'two_layer' or model == 'two_layer_nowriter' or model == 'two_layer_noeraser':
                # background phospho rate for second kinase/ first substrate
                if 'kin2_bg_phospho_rate' not in prefit_params and 'kin2_bg_phospho_rate' not in param_to_index:
                    param_to_index['kin2_bg_phospho_rate'] = len(x0)
                    x0.append(-2.0)
                    bounds.append((-8, 1))

                dataset_to_params[exp_name].append('kin2_bg_phospho_rate')
                
                # background phospho rate for second substrate
                if 'sub2_bg_phospho_rate' not in prefit_params and 'sub2_bg_phospho_rate' not in param_to_index:
                    param_to_index['sub2_bg_phospho_rate'] = len(x0)
                    x0.append(-2.0)
                    bounds.append((-8, 1))

                dataset_to_params[exp_name].append('sub2_bg_phospho_rate')
                
                
        # kinase parameters  
        if model == 'push' or model == 'pushpull' or model == 'two_layer' or model == 'two_layer_noeraser':
            
            # assign kinase phospho rate
            label = row['kinase_variant']
            if label not in prefit_params and label not in param_to_index:
                param_to_index[label] = len(x0)
                x0.append(0.0)
                bounds.append((-8, 8))

            dataset_to_params[exp_name].append(label)

            # assign kinase zipper binding affinity
            label = row['kinase_zipper']
            if label not in prefit_params and label not in param_to_index:
                param_to_index[label] = len(x0)
                x0.append(3.0)
                bounds.append((-8, 8))

            dataset_to_params[exp_name].append(label)
            
        # pptase parameters
        if model == 'pushpull' or model == 'two_layer' or model == 'two_layer_nowriter':
            
            # assign pptase phospho rate
            label = row['pptase_variant']
            if label not in prefit_params and label not in param_to_index:
                param_to_index[label] = len(x0)
                x0.append(0.0)
                bounds.append((-8, 4))

            dataset_to_params[exp_name].append(label)

            # assign pptase zipper binding affinity
            label = row['pptase_zipper']
            if label not in prefit_params and label not in param_to_index:
                param_to_index[label] = len(x0)
                x0.append(3.0)
                bounds.append((-8, 8))

            dataset_to_params[exp_name].append(label)
           
        # second kinase / first substrate params
        if model == 'two_layer' or model == 'two_layer_nowriter' or model == 'two_layer_noeraser':
            
            # assign kinase phospho rate
            label = row['kinase2_variant']
            if label not in prefit_params and label not in param_to_index:
                param_to_index[label] = len(x0)
                x0.append(0.0)
                bounds.append((-8, 8))

            dataset_to_params[exp_name].append(label)

            # assign kinase zipper binding affinity
            label = row['kinase2_zipper']
            if label not in prefit_params and label not in param_to_index:
                param_to_index[label] = len(x0)
                x0.append(3.0)
                bounds.append((-8, 8))

            dataset_to_params[exp_name].append(label)
            
            
            
    return  prefit_params, param_to_index, dataset_to_params, x0, bounds
        
def predict(x, args, df_copy):
        
    (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params, noise_models) = args
    
#     print(x)

    df_copy['phospho_MOCU_predict'] = 0.0
    df_copy['kinase2_phospho_MOCU_predict'] = 0.0
    
    for exp_name, row in df_dataset_key.iterrows():
        
#         print(exp_name)

        params = []
        for p in dataset_to_params[exp_name]:
            if p in prefit_params:
                params.append(prefit_params[p])
            else:
                params.append(x[param_to_index[p]])
        
#         print(exp_name, params)
        
        params = 10.0**np.array(params)
        
        df_tmp = df_copy.query("exp_name==@exp_name")
        
#         print(exp_name, len(df_tmp.index))

        if row['model'] == 'substrate_only':

            df_copy.loc[df_tmp.index, 'phospho_MOCU_predict'] = thermo.predict_substrate_only(df_tmp['substrate_MOCU_infer'].values, *params)
            df_copy.loc[df_tmp.index, 'kinase2_phospho_MOCU_predict'] = 0.0

        elif row['model'] == 'non-pplatable':

            df_copy.loc[df_tmp.index, 'phospho_MOCU_predict'] = thermo.predict_nonpplatable(df_tmp['substrate_MOCU_infer'].values, *params)
            df_copy.loc[df_tmp.index, 'kinase2_phospho_MOCU_predict'] = 0.0

        elif row['model'] == 'push':

            df_copy.loc[df_tmp.index, 'phospho_MOCU_predict'] = thermo.predict_push(df_tmp['kinase_MOCU_infer'].values, df_tmp['substrate_MOCU_infer'].values, *params)
            df_copy.loc[df_tmp.index, 'kinase2_phospho_MOCU_predict'] = 0.0

        elif row['model'] == 'pushpull':
            df_copy.loc[df_tmp.index, 'phospho_MOCU_predict'] = thermo.predict_pushpull(df_tmp['kinase_MOCU_infer'].values, df_tmp['pptase_MOCU_infer'].values, df_tmp['substrate_MOCU_infer'].values, *params)
            df_copy.loc[df_tmp.index, 'kinase2_phospho_MOCU_predict'] = 0.0
            
        elif row['model'] == 'two_layer':
            
            kinase2_phospho_MOCU_predict, phospho_MOCU_predict = thermo.predict_twolayer(df_tmp['kinase_MOCU_infer'].values, df_tmp['pptase_MOCU_infer'].values, df_tmp['kinase2_GFP_infer'].values, df_tmp['substrate_MOCU_infer'].values, *params)
            df_copy.loc[df_tmp.index, 'phospho_MOCU_predict'] = phospho_MOCU_predict
            df_copy.loc[df_tmp.index, 'kinase2_phospho_MOCU_predict'] = kinase2_phospho_MOCU_predict
            
        elif row['model'] == 'two_layer_nowriter':
            
            kinase2_phospho_MOCU_predict, phospho_MOCU_predict = thermo.predict_twolayer_nowriter(df_tmp['pptase_MOCU_infer'].values, df_tmp['kinase2_GFP_infer'].values, df_tmp['substrate_MOCU_infer'].values, *params)
            df_copy.loc[df_tmp.index, 'phospho_MOCU_predict'] = phospho_MOCU_predict
            df_copy.loc[df_tmp.index, 'kinase2_phospho_MOCU_predict'] = kinase2_phospho_MOCU_predict
            
        elif row['model'] == 'two_layer_noeraser':
            
            kinase2_phospho_MOCU_predict, phospho_MOCU_predict = thermo.predict_twolayer_nowriter(df_tmp['kinase_MOCU_infer'].values, df_tmp['kinase2_GFP_infer'].values, df_tmp['substrate_MOCU_infer'].values, *params)
            df_copy.loc[df_tmp.index, 'phospho_MOCU_predict'] = phospho_MOCU_predict
            df_copy.loc[df_tmp.index, 'kinase2_phospho_MOCU_predict'] = kinase2_phospho_MOCU_predict
            
    
# def loss_lognormal(x, args):

    
#     (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params, cnoise) = args
    
    
#     df_copy = df_data.dropna().copy()

#     predict(x, args, df_copy)
        
    
#     loss = 0.0
#     norm = 0.0
#     for exp_name, row in df_dataset_key.iterrows():
#         df_exp = df_data.query("exp_name==@exp_name")
#         if len(df_exp.index) > 0:
#             loss += np.sum((np.log10(df_exp['phospho_GFP_predict'])-np.log10(df_exp['phospho_GFP_infer']))**2)
#             norm += np.sum((np.log10(df_exp['phospho_GFP_infer'])-np.mean(np.log10(df_exp['phospho_GFP_infer'])))**2)

#     df_data = df_data.query("kinase2_phospho_GFP_infer>0.0")
#     for exp_name, row in df_dataset_key.iterrows():
#         df_exp = df_data.query("exp_name==@exp_name")
#         if len(df_exp.index) > 0:
#             loss += np.sum((np.log10(df_exp['kinase2_phospho_GFP_predict'])-np.log10(df_exp['kinase2_phospho_GFP_infer']))**2)
#             norm += np.sum((np.log10(df_exp['kinase2_phospho_GFP_infer'])-np.mean(np.log10(df_exp['kinase2_phospho_GFP_infer'])))**2)


#     print(loss/norm)
        
#     return loss / norm
  
    
    
def loss_lognormal_bg(x, args):

    
    (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params, noise_models) = args
    
    
    df_copy = df_data.copy()

    predict(x, args, df_copy)
    
    params = []
    for p in dataset_to_params['noise_model']:
        if p in prefit_params:
            params.append(prefit_params[p])
        else:
            params.append(x[param_to_index[p]])
                    
        
    loss = 0.0
    norm = 0.0
    for exp_name, row in df_dataset_key.iterrows():
        
            # if phospho factor exists, then multiply prediction

        if 'phospho_factor' in row.index.values:
            phospho_factor = row['substrate_phospho_factor']
        else:
            phospho_factor = 1.0
            
        if 'kinase2_phospho_factor' in row.index.values:
            kinase2_phospho_factor = row['kinase2_phospho_factor']
        else:
            kinase2_phospho_factor = 1.0
            
        
        df_exp = df_copy.query("exp_name==@exp_name")
        if len(df_exp.index) > 0:
            loss += loglikelihood_lognormal_bg(df_exp['phospho_GFP_infer'].values, phospho_factor*df_exp['phospho_MOCU_predict'].values, noise_models['phospho']['GFP2MOCU'], params[0])
    
        df_exp = df_exp.query("kinase2_phospho_GFP_infer>0.0")
        if len(df_exp.index) > 0:
            loss += loglikelihood_lognormal_bg(df_exp['kinase2_phospho_GFP_infer'].values, kinase2_phospho_factor*df_exp['kinase2_phospho_MOCU_predict'].values, noise_models['kinase2_phospho']['GFP2MOCU'], params[1])
   
    
    print(loss, norm)
        
    return loss
          
    
    
    
def loglikelihood_lognormal_bg(meas, predict, cnoise, sigma):
    


    p = cnoise.calc_prob_meas(meas, predict, sigma) 

    l = -np.sum(np.log(p + 1e-8)) / len(meas)

#     print(l, sigma)

    return l
    
    
def fit(df_dataset_key, df_data, df_params=None, noise_models=None):
    
    prefit_params, param_to_index, dataset_to_params, x0, bounds = setup_model_params(df_dataset_key, df_params=df_params, noise_models=noise_models)
    
    print(prefit_params)
    print(param_to_index)
    print(dataset_to_params)
    print(x0)
    print(bounds)
    
    if len(param_to_index) == 0:
        return df_params
    
    args = (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params, noise_models)
    
    def callback(x):
        print("#############################################################")
#         print("Loss:", loss(x, args))

        for p in param_to_index:
            print(p, x[param_to_index[p]], x0[param_to_index[p]])

        end = time.time()

        print("Total Time Elapsed", (end-start)/60, "minutes")
    
    start = time.time()
 
              
    res = opt.minimize(loss_lognormal_bg, x0, args=(args, ), method='L-BFGS-B', 
                               jac=None, bounds=bounds, 
                               options={'iprint':1, 'gtol': 1e-4, 'ftol':1e-4},
                              callback=callback)
              
              
    print(res)
    
    variance = res.fun
    
    s_list = []
    for p in param_to_index:
        s_list.append([p, res.x[param_to_index[p]], 0.0, 0.0])

    df_params = pd.concat([df_params, pd.DataFrame(s_list, columns=['name', 'val', 'val_min', 'val_max'])])
    
    return df_params


def calc_error(df_dataset_key, df_data, df_params, tol=0.01, noise_models=None):
    
    prefit_params, param_to_index, dataset_to_params, x0, bounds = setup_model_params(df_dataset_key, df_params=df_params.query("val_min != 0.0 and val_max != 0.0"), noise_models=noise_models)
    
    print(prefit_params)
    print(param_to_index)
    print(dataset_to_params)
    print(bounds)
    
    args = (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params, noise_models)
    
    for p in param_to_index:
        x0[param_to_index[p]] = df_params.query("name==@p").iloc[0]['val']
    
    print(x0)
    
    
    val_min = np.zeros_like(x0)
    val_max = np.zeros_like(x0)

    
    f0 = loss_lognormal_bg(x0, args)
    
    for p in param_to_index:

        i = param_to_index[p]

        print(i, p)

        def func(val):

            x = x0.copy()
            x[i] = val

            f = loss_lognormal_bg(x, args) - (1+tol)*f0

            print(f, val)

            return f

        print("min", x0[i])
        fmin = func(bounds[i][0])

        if fmin < 0.0:
            val_min[i] = bounds[i][0]
        else:
            res = opt.root_scalar(func, bracket=[bounds[i][0], x0[i]], method='bisect', xtol=1e-4)

            print(res)

            val_min[i] = res.root

        print("max", x0[i])
        fmax = func(bounds[i][1])

        if fmax < 0.0:
            val_max[i] = bounds[i][1]
        else:
            res = opt.root_scalar(func, bracket=[x0[i], bounds[i][1]], method='brentq', xtol=1e-4)

            print(res)

            val_max[i] = res.root

        print(i, p, val_min[i], x0[i], val_max[i])
        
        df_params.loc[df_params.name==p, 'val_min'] = val_min[i]
        df_params.loc[df_params.name==p, 'val_max'] = val_max[i]
    

    return df_params


def fit_bg_noise(meas, cnoise):


    def loss(x, args):

        (mu, sigma) = x.tolist()
        c0 = np.exp(mu)

        meas = args


        p = cnoise.calc_prob_meas(meas, c0*np.ones_like(meas), sigma) 

        l = -np.sum(np.log(p + 1e-8)) / len(meas)

        print(l, c0, sigma)

        return l
    
    
    x0 = np.array([np.log(1e3), 1.0])
    bounds = [(np.log(1e-8), np.log(1e6)), (0.1, 2.0)]
    res = opt.minimize(loss, x0, args=(meas,), method='L-BFGS-B', bounds=bounds, 
                       options={'iprint':1, 'gtol': 1e-4, 'ftol':1e-4})
        
    print(res)
    
    return res.x

def fit_phospho_bg_noise(meas, predict, cnoise):


    def loss(x, args):

        sigma = x[0]

        (meas, predict) = args


        p = cnoise.calc_prob_meas(meas, predict, sigma) 

        l = -np.sum(np.log(p + 1e-8)) / len(meas)

        print(l, sigma)

        return l
    
    
    x0 = np.array([1.0])
    bounds = [(0.1, 2.0)]
    res = opt.minimize(loss, x0, args=((meas, predict), ), method='L-BFGS-B', bounds=bounds, 
                       options={'iprint':1, 'gtol': 1e-4, 'ftol':1e-4})
        
    print(res)
    
    return res.x
    