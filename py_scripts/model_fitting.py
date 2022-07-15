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



def setup_model_params(df_dataset_key, df_params=None):
    

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
    
    
    for exp_name, row in df_dataset_key.iterrows():
    
    
        dataset_to_params[exp_name] = []
    
        model = row['model']
        if model == 'non-pplatable':
            continue

        # initialize parameters used in every model
        if 'bg_phospho_rate' not in prefit_params and 'bg_phospho_rate' not in param_to_index:
            param_to_index['bg_phospho_rate'] = len(x0)
            x0.append(-2.0)
            bounds.append((-8, 1))
    
        # parameters used in every model
        dataset_to_params[exp_name].append('bg_phospho_rate')
            
        if model == 'push' or model == 'pushpull' or model == 'two_layer':
            
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
            
        if model == 'pushpull' or model == 'two_layer':
            
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
            
        if model == 'two_layer':
            
            # assign kinase phospho rate
            label = row['kinase2_variant']
            if label not in prefit_params and label not in param_to_index:
                param_to_index[label] = len(x0)
                x0.append(0.0)
                bounds.append((-8, 8))

            dataset_to_params[exp_name].append(label)

            # assign kinase zipper binding affinity
            label = row['SH2_binding']
            if label not in prefit_params and label not in param_to_index:
                param_to_index[label] = len(x0)
                x0.append(1.0)
                bounds.append((-8, 8))

            dataset_to_params[exp_name].append(label)
            
            # background phospho and dephospho rates for second kinase
            if 'bg_kinase2phospho_rate' not in prefit_params and 'bg_kinase2phospho_rate' not in param_to_index:
                param_to_index['bg_kinase2phospho_rate'] = len(x0)
                x0.append(-2.0)
                bounds.append((-8, 1))
                
            if 'bg_kinase2dephospho_rate' not in prefit_params and 'bg_kinase2dephospho_rate' not in param_to_index:
                param_to_index['bg_kinase2dephospho_rate'] = len(x0)
                x0.append(-2.0)
                bounds.append((-8, 1))
    
            dataset_to_params[exp_name].insert(0, 'bg_kinase2phospho_rate')
            dataset_to_params[exp_name].insert(1, 'bg_kinase2dephospho_rate')
            
    return  prefit_params, param_to_index, dataset_to_params, x0, bounds
        
def predict(x, args, df_copy):
    
    (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params) = args
    
    for exp_name, row in df_dataset_key.iterrows():

        params = []
        for p in dataset_to_params[exp_name]:
            if p in prefit_params:
                params.append(prefit_params[p])
            else:
                params.append(x[param_to_index[p]])
        
        params = 10.0**np.array(params)

        df_tmp = df_copy.query("exp_name==@exp_name")
        
#         print(exp_name, len(df_tmp.index))

        if row['model'] == 'substrate_only':

            df_copy.loc[df_tmp.index, 'phospho_GFP_predict'] = thermo.predict_substrate_only(df_tmp['substrate_GFP_infer'].values, *params)
            df_copy.loc[df_tmp.index, 'kinase2phospho_GFP_predict'] = 0.0

        elif row['model'] == 'non-pplatable':

            df_copy.loc[df_tmp.index, 'phospho_GFP_predict'] = thermo.predict_nonpplatable(df_tmp['substrate_GFP_infer'].values)
            df_copy.loc[df_tmp.index, 'kinase2phospho_GFP_predict'] = 0.0

        elif row['model'] == 'push':

            df_copy.loc[df_tmp.index, 'phospho_GFP_predict'] = thermo.predict_push(df_tmp['kinase_GFP_infer'].values, df_tmp['substrate_GFP_infer'].values, *params)
            df_copy.loc[df_tmp.index, 'kinase2phospho_GFP_predict'] = 0.0

        elif row['model'] == 'pushpull':
            df_copy.loc[df_tmp.index, 'phospho_GFP_predict'] = thermo.predict_pushpull(df_tmp['kinase_GFP_infer'].values, df_tmp['pptase_GFP_infer'].values, df_tmp['substrate_GFP_infer'].values, *params)
            df_copy.loc[df_tmp.index, 'kinase2phospho_GFP_predict'] = 0.0
            
        elif row['model'] == 'two_layer':
            
            kinase2phospho_GFP_predict, phospho_GFP_predict = thermo.predict_twolayerpushpull(df_tmp['kinase_GFP_infer'].values, df_tmp['pptase_GFP_infer'].values, df_tmp['kinase2_GFP_infer'].values, df_tmp['substrate_GFP_infer'].values, *params)
            df_copy.loc[df_tmp.index, 'phospho_GFP_predict'] = phospho_GFP_predict
            df_copy.loc[df_tmp.index, 'kinase2phospho_GFP_predict'] = kinase2phospho_GFP_predict
            
    
def loss(x, args):

    
    (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params) = args
    
    
    df_copy = df_data.dropna().copy()

    predict(x, args, df_copy)
    
  
    
    return loss_func(df_copy, df_dataset_key)
    

    
def loss_func(df_data, df_dataset_key):
    
    loss1 = 0.0
    N = 0
    for exp_name, row in df_dataset_key.iterrows():
        df_exp = df_data.query("exp_name==@exp_name")
        if len(df_exp.index) > 0:
            loss1 += np.sum((np.log10(df_exp['phospho_GFP_predict'])-np.log10(df_exp['phospho_GFP_infer']))**2)
#             loss1 += np.sum((df_exp['phospho_GFP_predict']/df_exp['substrate_GFP_infer']-df_exp['phospho_frac_infer'])**2)
            
        N += len(df_exp.index)
        
    loss1 /= N
            
    loss2 = 0.0
    N = 0
    df_data = df_data.query("kinase2phospho_GFP_infer>0.0")
    for exp_name, row in df_dataset_key.iterrows():
        df_exp = df_data.query("exp_name==@exp_name")
        if len(df_exp.index) > 0:
            loss2 += np.sum((np.log10(df_exp['kinase2phospho_GFP_predict'])-np.log10(df_exp['kinase2phospho_GFP_infer']))**2)
#             loss2 += np.sum((df_exp['kinase2phospho_GFP_predict']/df_exp['substrate_GFP_infer']-df_exp['kinase2phospho_frac_infer'])**2)
        N += len(df_exp.index)
    
    if N > 0:
        loss2 /= N
        
    print(loss1+loss2)
        
    return loss1 + loss2
    
    
    
    
    
def fit(df_dataset_key, df_data, df_params=None):
    
    prefit_params, param_to_index, dataset_to_params, x0, bounds = setup_model_params(df_dataset_key, df_params=df_params)
    
    print(prefit_params)
    print(param_to_index)
    print(dataset_to_params)
    print(x0)
    print(bounds)
    
    if len(param_to_index) == 0:
        return df_params
    
    args = (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params)
    
    def callback(x):
        print("#############################################################")
#         print("Loss:", loss(x, args))

        for p in param_to_index:
            print(p, x[param_to_index[p]], x0[param_to_index[p]])

        end = time.time()

        print("Total Time Elapsed", (end-start)/60, "minutes")
    
    start = time.time()
    
#     optimize_SGD(loss, x0, args=args, callback=callback)
    
    
              
    res = opt.minimize(loss, x0, args=(args, ), method='L-BFGS-B', 
                               jac='2-point', bounds=bounds, 
                               options={'iprint':1, 'eps': 1e-6, 
                                        'gtol': 1e-4, 'ftol':1e-4},
                              callback=callback)
              
              
    print(res)
    
    variance = res.fun
    
    s_list = []
    for p in param_to_index:
        s_list.append([p, res.x[param_to_index[p]], 0.0, 0.0])

    df_params = pd.concat([df_params, pd.DataFrame(s_list, columns=['name', 'val', 'val_min', 'val_max'])])
    
    return df_params




def calc_error(df_dataset_key, df_data, df_params, tol=0.01):
    
    prefit_params, param_to_index, dataset_to_params, x0, bounds = setup_model_params(df_dataset_key, df_params=df_params.query("val_min != 0.0 and val_max != 0.0"))
    
    print(prefit_params)
    print(param_to_index)
    print(dataset_to_params)
    print(bounds)
    
    args = (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params)
    
    for p in param_to_index:
        x0[param_to_index[p]] = df_params.query("name==@p").iloc[0]['val']
    
    print(x0)
    
    
    val_min = np.zeros_like(x0)
    val_max = np.zeros_like(x0)

    
    f0 = loss(x0, args)
    
    for p in param_to_index:

        i = param_to_index[p]

        print(i, p)

        def func(val):

            x = x0.copy()
            x[i] = val

            f = loss(x, args) - (1+tol)*f0

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
    