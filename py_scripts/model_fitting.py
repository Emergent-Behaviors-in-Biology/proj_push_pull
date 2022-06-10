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
                x0.append(-1.0)
                bounds.append((-8, 4))

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
                x0.append(-1.0)
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
                x0.append(-1.0)
                bounds.append((-8, 8))

            dataset_to_params[exp_name].append(label)

            # assign kinase zipper binding affinity
            label = row['kinase2_zipper']
            if label not in prefit_params and label not in param_to_index:
                param_to_index[label] = len(x0)
                x0.append(1.0)
                bounds.append((-8, 8))

            dataset_to_params[exp_name].append(label)
            
            # background phospho and dephoshp rates for second kinase
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
    
    (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params, phospho_GFP_cutoff) = args
    
    for exp_name, row in df_dataset_key.iterrows():

        params = []
        for p in dataset_to_params[exp_name]:
            if p in prefit_params:
                params.append(prefit_params[p])
            else:
                params.append(x[param_to_index[p]])
        
        params = 10.0**np.array(params)

        df_tmp = df_copy.query("exp_name==@exp_name")

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

    
    (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params, phospho_GFP_cutoff) = args
    
    
    df_copy = df_data.dropna().copy()

    predict(x, args, df_copy)
    
#     loss = 0.0

#     for exp_name, row in df_dataset_key.iterrows():
#         loss += np.mean((np.log10(df_copy.query("exp_name==@exp_name")['phospho_GFP_predict'])-np.log10(df_copy.query("exp_name==@exp_name")['phospho_GFP_infer']))**2)

    loss = np.mean((np.log10(df_copy['phospho_GFP_predict'])-np.log10(df_copy['phospho_GFP_infer']))**2)
    
    df_copy = df_copy.query("kinase2phospho_GFP_infer>0.0")
    
    loss +=np.mean((np.log10(df_copy['kinase2phospho_GFP_predict'])-np.log10(df_copy['kinase2phospho_GFP_infer']))**2)
    
    
    return loss
    

def fit(df_dataset_key, df_data, phospho_GFP_cutoff, df_params=None):
    
    prefit_params, param_to_index, dataset_to_params, x0, bounds = setup_model_params(df_dataset_key, df_params=df_params)
    
    print(prefit_params)
    print(param_to_index)
    print(dataset_to_params)
    print(x0)
    print(bounds)
    
    if len(param_to_index) == 0:
        return df_params
    
    args = (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params, phospho_GFP_cutoff)
    
    def callback(x):
        print("#############################################################")
        print("Loss:", loss(x, args))

        for p in param_to_index:
            print(p, x[param_to_index[p]], x0[param_to_index[p]])

        end = time.time()

        print("Total Time Elapsed", (end-start)/60, "minutes")
    
    start = time.time()
              
    res = opt.minimize(loss, x0, args=(args, ), method='L-BFGS-B', 
                               jac='2-point', bounds=bounds, 
                               options={'iprint':1, 'eps': 1e-6, 
                                        'gtol': 1e-4, 'ftol':1e-4},
                              callback=callback)
              
              
    print(res)
    
    s_list = []
    for p in param_to_index:
        s_list.append([p, res.x[param_to_index[p]], 0.0, 0.0])

    df_params = pd.concat([df_params, pd.DataFrame(s_list, columns=['name', 'val', 'val_min', 'val_max'])])
    
    return df_params

def calc_error(df_dataset_key, df_data, phospho_GFP_cutoff, df_params, tol=0.01):
    
    prefit_params, param_to_index, dataset_to_params, x0, bounds = setup_model_params(df_dataset_key, df_params=df_params.query("val_min != 0.0 and val_max != 0.0"))
    
    print(prefit_params)
    print(param_to_index)
    print(dataset_to_params)
    print(bounds)
    
    args = (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params, phospho_GFP_cutoff)
    
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
    

# def setup_model_params(df_dataset_key):
    
    
#     # map of parameters names to indices
#     param_dict = {'bg_phospho_rate': 0}
#     # map of datasets to lists of relevant parameters
#     model_params = {}

#     # list of initial conditions
#     x0 = [-2.0]
#     # list of parameter bounds
#     bounds = [(-8, 1)]


#     param_index = 1
#     for exp_name, row in df_dataset_key.iterrows():

#         model = row['model']
        
#         if model == 'substrate_only':
#             # assign background phospho rate
#             model_params[exp_name] = [param_dict['bg_phospho_rate']]
            
#         elif model == 'non-pplatable':
#             # no parameters
#             model_params[exp_name] = []

#         elif model == 'push':

#             # assign background phospho rate
#             model_params[exp_name] = [param_dict['bg_phospho_rate']]

#             # assign kinase phospho rate
#             kinase = row['kinase_variant']
#             if kinase not in param_dict:
#                 param_dict[kinase] = len(param_dict)
#                 x0.append(-1.0)
#                 bounds.append((-8, 2))

#             model_params[exp_name].append(param_dict[kinase])

#             # assign kinase zipper binding affinity
#             zipper = row['kinase_zipper']
#             if zipper not in param_dict:
#                 param_dict[zipper] = len(param_dict)
#                 x0.append(3.0)
#                 bounds.append((-8, 8))

#             model_params[exp_name].append(param_dict[zipper])
            
#         elif model == 'pushpull':

#             # assign background phospho rate
#             model_params[exp_name] = [param_dict['bg_phospho_rate']]

#             # assign kinase phospho rate
#             kinase = row['kinase_variant']
#             if kinase not in param_dict:
#                 param_dict[kinase] = len(param_dict)
#                 x0.append(-1.0)
#                 bounds.append((-8, 2))

#             model_params[exp_name].append(param_dict[kinase])

#             # assign kinase zipper binding affinity
#             zipper = row['kinase_zipper']
#             if zipper not in param_dict:
#                 param_dict[zipper] = len(param_dict)
#                 x0.append(3.0)
#                 bounds.append((-8, 8))

#             model_params[exp_name].append(param_dict[zipper])
            
#             # assign pptase phospho rate
#             pptase = row['pptase_variant']
#             if pptase not in param_dict:
#                 param_dict[pptase] = len(param_dict)
#                 x0.append(-1.0)
#                 bounds.append((-8, 2))

#             model_params[exp_name].append(param_dict[pptase])

#             # assign pptase zipper binding affinity
#             zipper = row['pptase_zipper']
#             if zipper not in param_dict:
#                 param_dict[zipper] = len(param_dict)
#                 x0.append(3.0)
#                 bounds.append((-8, 8))

#             model_params[exp_name].append(param_dict[zipper])


#     return param_dict, model_params, x0, bounds
            

# def fit(df_dataset_key, df_data, phospho_GFP_cutoff, alpha=1e-4):


#     def solve(df_dataset_key, df_data, model_params, param_dict, x0, bounds, verbose=False):

#         df_copy = df_data.dropna().copy()

#         if verbose:
#             start = time.time()

#     #     loss_dict = {}
#         def func(x):

#             loss = 0.0

#             for exp_name, row in df_dataset_key.iterrows():

                
#                 params = 10**np.array(x)[model_params[exp_name]]

#                 df_tmp = df_copy.query("exp_name==@exp_name")

#                 if row['model'] == 'substrate_only':
                    
#                     df_copy.loc[df_tmp.index, 'phospho_conc_predict'] = thermo.predict_substrate_only(df_tmp['substrate_conc_infer'].values, *params)
                                        
#                 elif row['model'] == 'non-pplatable':
                    
#                     df_copy.loc[df_tmp.index, 'phospho_conc_predict'] = thermo.predict_nonpplatable(df_tmp['substrate_conc_infer'].values)
                    
#                 elif row['model'] == 'push':

#                     df_copy.loc[df_tmp.index, 'phospho_conc_predict'] = thermo.predict_push(df_tmp['kinase_conc_infer'].values, df_tmp['substrate_conc_infer'].values, *params)

#                 elif row['model'] == 'pushpull':
#                     df_copy.loc[df_tmp.index, 'phospho_conc_predict'] = thermo.predict_pushpull(df_tmp['kinase_conc_infer'].values, df_tmp['pptase_conc_infer'].values, df_tmp['substrate_conc_infer'].values, *params)

                    
                

#                 df_copy.loc[df_tmp.index, 'phospho_GFP_predict'] = df_copy.loc[df_tmp.index, 'phospho_conc_predict'] + phospho_GFP_cutoff

                
# #                 MSE = np.mean((np.log10(df_copy.loc[df_tmp.index, 'phospho_GFP_predict'])-np.log10(df_copy.loc[df_tmp.index, 'phospho_GFP_infer']))**2)
# #                 loss += MSE 
                

#             loss = np.mean((np.log10(df_copy['phospho_GFP_predict'])-np.log10(df_copy['phospho_GFP_infer']))**2)
                
                
                
                
#             # add small tether regularization to initial conditions
# #             loss += alpha*np.sum((x-np.array(x0))**2)

#             return loss




        

#         def callback(x):
#             print("#############################################################")
#             print("Total Loss:", func(x), "Regularization:", alpha*np.sum((x-np.array(x0))**2))
            
#             for p in param_dict:
#                 print(p, x0[param_dict[p]], x[param_dict[p]])
                         
#             end = time.time()

#             print("Total Time Elapsed", (end-start)/60, "minutes")

            
#         callback(x0)
            

#         res = opt.minimize(func, x0, method='L-BFGS-B', 
#                            jac='2-point', bounds=bounds, 
#                            options={'iprint':1, 'eps': 1e-6, 
#                                     'gtol': 1e-4, 'ftol':1e-8},
#                           callback=callback)
# #         res = opt.minimize(func, x0, method='BFGS', 
# #                            jac='2-point', 
# #                            options={'eps': 1e-6, 'gtol': 1e-4, 'disp': True},
# #                           callback=callback)


#         callback(res.x)

#         print(res)


#         return res

#     param_dict, model_params, x0, bounds = setup_model_params(df_dataset_key)


#     print(param_dict)
#     print(model_params)
#     print(x0)
#     print(bounds)

#     res = solve(df_dataset_key, df_data, model_params, param_dict, x0, bounds, verbose=True)

    
#     for exp_name, row in df_dataset_key.iterrows():
    
#         model = row['model']
        
#         params = 10**res.x[model_params[exp_name]]
        
#         if model == 'substrate_only':
#             df_dataset_key.loc[exp_name, 'bg_phospho_rate'] = params[0]
#         elif model == 'push':
#             df_dataset_key.loc[exp_name, 'bg_phospho_rate'] = params[0]
#             df_dataset_key.loc[exp_name, 'kinase_phospho_rate'] = params[1]
#             df_dataset_key.loc[exp_name, 'kinase_binding_affinity'] = params[2]
#         elif model == 'pushpull':
#             df_dataset_key.loc[exp_name, 'bg_phospho_rate'] = params[0]
#             df_dataset_key.loc[exp_name, 'kinase_phospho_rate'] = params[1]
#             df_dataset_key.loc[exp_name, 'kinase_binding_affinity'] = params[2]
#             df_dataset_key.loc[exp_name, 'pptase_dephospho_rate'] = params[3]
#             df_dataset_key.loc[exp_name, 'pptase_binding_affinity'] = params[4]
              
        
        
#     display(df_dataset_key)
    
#     return res, param_dict

# def calc_error(df_dataset_key, df_data, phospho_GFP_cutoff, tol=0.01, alpha=1e-4):

    
#     def solve(df_dataset_key, df_data, model_params, param_dict, x, x0, bounds, verbose=False):

#         df_copy = df_data.dropna().copy()

#         if verbose:
#             start = time.time()

#     #     loss_dict = {}
#         def loss(x):

#             loss = 0.0



#             for exp_name, row in df_dataset_key.iterrows():

                
#                 params = 10**np.array(x)[model_params[exp_name]]

#                 df_tmp = df_copy.query("exp_name==@exp_name")

#                 if row['model'] == 'substrate_only':
                    
#                     df_copy.loc[df_tmp.index, 'phospho_conc_predict'] = thermo.predict_substrate_only(df_tmp['substrate_conc_infer'].values, *params)
                                        
#                 elif row['model'] == 'non-pplatable':
                    
#                     df_copy.loc[df_tmp.index, 'phospho_conc_predict'] = thermo.predict_nonpplatable(df_tmp['substrate_conc_infer'].values)
                    
#                 elif row['model'] == 'push':

#                     df_copy.loc[df_tmp.index, 'phospho_conc_predict'] = thermo.predict_push(df_tmp['kinase_conc_infer'].values, df_tmp['substrate_conc_infer'].values, *params)

#                 elif row['model'] == 'pushpull':
#                     df_copy.loc[df_tmp.index, 'phospho_conc_predict'] = thermo.predict_pushpull(df_tmp['kinase_conc_infer'].values, df_tmp['pptase_conc_infer'].values, df_tmp['substrate_conc_infer'].values, *params)

                    
                

#                 df_copy.loc[df_tmp.index, 'phospho_GFP_predict'] = df_copy.loc[df_tmp.index, 'phospho_conc_predict'] + phospho_GFP_cutoff

                
# #                 MSE = np.mean((np.log10(df_copy.loc[df_tmp.index, 'phospho_GFP_predict'])-np.log10(df_copy.loc[df_tmp.index, 'phospho_GFP_infer']))**2)
# #                 var = np.mean((np.log10(df_copy.loc[df_tmp.index, 'phospho_GFP_infer'])-np.log10(df_copy.loc[df_tmp.index, 'phospho_GFP_infer']).mean())**2)
                
# #                 loss += MSE 

#             loss = np.mean((np.log10(df_copy['phospho_GFP_predict'])-np.log10(df_copy['phospho_GFP_infer']))**2)
                
#             return loss



#         y_low = np.zeros(len(param_dict))
#         y_up = np.zeros(len(param_dict))
        

#         f0 = loss(x)
# #         print(f0, alpha*np.sum((x-np.array(x0))**2))
# #         f0 += alpha*np.sum((x-np.array(x0))**2)
        
#         for p in param_dict:
            
#             i = param_dict[p]
            
#             print(i, p)
            
#             def func(y):
                
#                 X = x.copy()
#                 X[i] = y
                
#                 f = loss(X) - (1+tol)*f0
                
#                 print(f, y)
                
#                 return f
            
#             print("min", x[i])
#             fmin = func(bounds[i][0])
            
#             if fmin < 0.0:
#                 y_low[i] = bounds[i][0]
#             else:
#                 res = opt.root_scalar(func, bracket=[bounds[i][0], x[i]], method='bisect', xtol=1e-4)
            
#                 print(res)
             
#                 y_low[i] = res.root
               
#             print("max", x[i])
#             fmax = func(bounds[i][1])
                
#             if fmax < 0.0:
#                 y_up[i] = bounds[i][1]
#             else:
#                 res = opt.root_scalar(func, bracket=[x[i], bounds[i][1]], method='brentq', xtol=1e-4)
            
#                 print(res)
             
#                 y_up[i] = res.root
                
#             print(i, p, y_low[i], x[i], y_up[i])
                
#         return (y_low, y_up)

#     param_dict, model_params, x0, bounds = setup_model_params(df_dataset_key)


#     print(param_dict)
#     print(model_params)
#     print(x0)
#     print(bounds)
    
    
#     x = np.zeros(len(param_dict))

#     for exp_name, row in df_dataset_key.iterrows():

#         model = row['model']

#         params = np.zeros(len(model_params[exp_name]))

#         if model == 'substrate_only':
#             params[0] = df_dataset_key.loc[exp_name, 'bg_phospho_rate']
#         elif model == 'push':
#             params[0] = df_dataset_key.loc[exp_name, 'bg_phospho_rate']
#             params[1] = df_dataset_key.loc[exp_name, 'kinase_phospho_rate']
#             params[2] = df_dataset_key.loc[exp_name, 'kinase_binding_affinity']
#         elif model == 'pushpull':
#             params[0] = df_dataset_key.loc[exp_name, 'bg_phospho_rate']
#             params[1] = df_dataset_key.loc[exp_name, 'kinase_phospho_rate']
#             params[2] = df_dataset_key.loc[exp_name, 'kinase_binding_affinity']
#             params[3] = df_dataset_key.loc[exp_name, 'pptase_dephospho_rate']
#             params[4] = df_dataset_key.loc[exp_name, 'pptase_binding_affinity']

#         x[model_params[exp_name]] = np.log10(params)
    

#     y_low, y_up = solve(df_dataset_key, df_data, model_params, param_dict, x, x0, bounds, verbose=True)

#     df_params = pd.DataFrame(np.c_[x, y_low, y_up], columns=['val', 'low', 'high'])
#     params = ['']*len(param_dict)
#     for p in param_dict:
#         params[param_dict[p]] = p
#     df_params['params'] = params
    
#     df_params.set_index('params', inplace=True)
      
#     return df_params
    

def calc_error_ABCMCMC(df_dataset_key, df_data, phospho_GFP_cutoff, df_param_dist=None, N_iters=10, tol=0.01, lamb=1e-4):

    def solve(df_dataset_key, df_data, model_params, param_dict, bounds, N_iters, tol, x0, start_iter=0, params=None):

        df_copy = df_data.dropna().copy()

        max_attempts = 100

        scale = 0.05*np.ones(len(param_dict))
        
        def prob(p, x0, scale):
            return np.exp(-0.5*np.sum((p-x0)**2*lamb+np.log(2*np.pi/lamb)))
    
        def kernel(pi, pj, scale):
            return np.exp(-0.5*np.sum((pj-pi)**2/scale**2+np.log(2*np.pi*scale**2)))
        
        def loss(x):
            loss = 0.0


            for exp_name, group in df_copy.groupby("exp_name"):

                p = 10**x[model_params[exp_name]]                

                if df_dataset_key.loc[exp_name ,'model'] == 'substrate_only':

                    df_copy.loc[group.index, 'phospho_conc_predict'] = thermo.predict_substrate_only(group['substrate_conc_infer'].values, *p)

                elif df_dataset_key.loc[exp_name ,'model'] == 'non-pplatable':

                    df_copy.loc[group.index, 'phospho_conc_predict'] = thermo.predict_nonpplatable(group['substrate_conc_infer'].values)

                elif df_dataset_key.loc[exp_name ,'model'] == 'push':

                    df_copy.loc[group.index, 'phospho_conc_predict'] = thermo.predict_push(group['kinase_conc_infer'].values, group['substrate_conc_infer'].values, *p)

                elif df_dataset_key.loc[exp_name ,'model'] == 'pushpull':
                    df_copy.loc[group.index, 'phospho_conc_predict'] = thermo.predict_pushpull(group['kinase_conc_infer'].values, group['pptase_conc_infer'].values, group['substrate_conc_infer'].values, *p)




                df_copy.loc[group.index, 'phospho_GFP_predict'] = df_copy.loc[group.index, 'phospho_conc_predict'] + phospho_GFP_cutoff


                MSE = np.mean((np.log10(df_copy.loc[group.index, 'phospho_GFP_predict'])-np.log10(df_copy.loc[group.index, 'phospho_GFP_infer']))**2)

                loss += MSE 
                
            return loss
        
      
            
        f0 = loss(x0)
        
        if start_iter == 0:
            params = x0
            error = f0
        else:
            error = loss(params)
            
        
        
        for t in range(start_iter, N_iters):
            
            rand.seed(t)

            print("iter:", t)

            candidate_params = rand.normal(loc=params, scale=scale)            

            candidate_error = loss(candidate_params)

            print(candidate_error, error, f0)
            
#             alpha = min(1, np.exp(-(candidate_error-error)/tol))
#             alpha = min(1, prob(candidate_params, x0, scale)/prob(params, x0, scale))

            alpha = min(1, np.exp(-(candidate_error+lamb*np.sum((candidate_params-x0)**2)-error-lamb*np.sum((params-x0)**2))/tol))
            
            
            print("alpha", alpha)
                
            p = rand.random()

            if p < alpha:
                print("accepted")
                params = candidate_params
                error = candidate_error
            
            
            
#             # if within error tolerance, accept
#             if candidate_error < (1+tol)*f0:
                
#                 print("within error tolerance")
                
#                 alpha = min(1, prob(candidate_params, x0, scale)/prob(params, x0, scale))
                
#                 print(prob(candidate_params, x0, scale), prob(params, x0, scale))
                
#                 print("alpha", alpha)
                
#                 p = rand.random()
                
#                 if p < alpha:
#                     print("accepted")
#                     params = candidate_params
#                     error = candidate_error
                
                
            yield t, params
            

    def callback(df_param_dist):
            print("#############################################################")

            for p in param_dict:
                print(p, np.mean(df_param_dist[p]), "+/-", np.var(df_param_dist[p]))
                         
            end = time.time()

            print("Total Time Elapsed", (end-start)/60, "minutes")
            
            print("#############################################################")
            
            scale = 0.1*np.ones(len(param_dict))
            
            ncols = 4
            nrows = int(np.ceil(len(param_dict)/ncols))
            
            fig = plt.figure(figsize=(4*ncols, 3*nrows))
                        
            for p in param_dict:
                
                i = param_dict[p]

                                
                ax = fig.add_subplot(nrows, ncols, i+1)
                
                ax.set_xlabel(p)
                
#                 (l, u) = bounds[i]
                l = np.min([np.min(df_param_dist[p]), x0[i]-4*scale[i]])
                u = np.max([np.max(df_param_dist[p]), x0[i]+4*scale[i]])
        
                t = np.linspace(l, u, 100)
                ax.plot(t, np.exp(-0.5*(t-x0[i])**2/scale[i]**2) / np.sqrt(2*np.pi*scale[i]**2), 'k--')
                                
                sns.histplot(x=df_param_dist[p], binrange=(l, u), ax=ax, stat='density')
                
                
#                 ax.vlines(x0[i])
                
                ax.set_xlim(l, u)

                
            plt.tight_layout()
            plt.show()
            
    
    param_dict, model_params, x0, bounds = setup_model_params(df_dataset_key)


    print(param_dict)
    print(model_params)
    print(bounds)
    
    
    if df_param_dist is None:
    
        x0 = np.zeros(len(param_dict))

        for exp_name, row in df_dataset_key.iterrows():

            model = row['model']

            params = np.zeros(len(model_params[exp_name]))

            if model == 'substrate_only':
                params[0] = df_dataset_key.loc[exp_name, 'bg_phospho_rate']
            elif model == 'push':
                params[0] = df_dataset_key.loc[exp_name, 'bg_phospho_rate']
                params[1] = df_dataset_key.loc[exp_name, 'kinase_phospho_rate']
                params[2] = df_dataset_key.loc[exp_name, 'kinase_binding_affinity']
            elif model == 'pushpull':
                params[0] = df_dataset_key.loc[exp_name, 'bg_phospho_rate']
                params[1] = df_dataset_key.loc[exp_name, 'kinase_phospho_rate']
                params[2] = df_dataset_key.loc[exp_name, 'kinase_binding_affinity']
                params[3] = df_dataset_key.loc[exp_name, 'pptase_dephospho_rate']
                params[4] = df_dataset_key.loc[exp_name, 'pptase_binding_affinity']

            x0[model_params[exp_name]] = np.log10(params)
            

        columns=['iter'] + len(param_dict)*['none']
        for p in param_dict:
            columns[1+param_dict[p]] = p

        df_param_dist = pd.DataFrame([], columns=columns)

        df_param_dist.set_index(['iter'], inplace=True)
        
        df_param_dist.loc[-1, :] = x0
                            
        display(df_param_dist)
        
        start_iter = 0
        params = None
        weights = None
    
    else:
        
        if 'iter' in df_param_dist.columns.values:
            df_param_dist.set_index(['iter'], inplace=True)
        
        x0 = df_param_dist.query("iter==-1").to_numpy().flatten()
        
        
        start_iter = df_param_dist.index.max()
        
        print(start_iter)
        
        params = df_param_dist.query("iter==@start_iter").to_numpy().flatten().astype(np.float64)
        
        start_iter += 1
              
    
    start = time.time()

    for (t, params) in solve(df_dataset_key, df_data, model_params, param_dict, bounds, N_iters=N_iters, tol=tol, x0=x0, start_iter=start_iter, params=params):
        
        df_param_dist.loc[t] = params
        
        if t % 100 == 0:
            callback(df_param_dist)
            
            display(df_param_dist)
        
    return df_param_dist


def calc_error_ABCSMC(df_dataset_key, df_data, phospho_GFP_cutoff, df_param_dist=None, N_replicas=100, N_iters=10, tol=0.01):

    def solve(df_dataset_key, df_data, model_params, param_dict, bounds, N_replicas, N_iters, tol, x0, start_iter=0, params=None, weights=None):

        df_copy = df_data.dropna().copy()

        max_attempts = 100
        
#         lower = np.array([x[0] for x in bounds])
#         upper = np.array([x[1] for x in bounds])
        
#         scale = (upper-lower)/100

        scale = 0.1*np.ones(len(param_dict))
        
        def prob(p, x0, scale):
            return np.exp(-0.5*np.sum((p-x0)**2/scale**2+np.log(2*np.pi*scale**2)))
    
        def kernel(pi, pj, scale):
            return np.exp(-0.5*np.sum((pj-pi)**2/scale**2+np.log(2*np.pi*scale**2)))
        
        def loss(x):
            loss = 0.0


            for exp_name, group in df_copy.groupby("exp_name"):

                p = 10**x[model_params[exp_name]]                

                if df_dataset_key.loc[exp_name ,'model'] == 'substrate_only':

                    df_copy.loc[group.index, 'phospho_conc_predict'] = thermo.predict_substrate_only(group['substrate_conc_infer'].values, *p)

                elif df_dataset_key.loc[exp_name ,'model'] == 'non-pplatable':

                    df_copy.loc[group.index, 'phospho_conc_predict'] = thermo.predict_nonpplatable(group['substrate_conc_infer'].values)

                elif df_dataset_key.loc[exp_name ,'model'] == 'push':

                    df_copy.loc[group.index, 'phospho_conc_predict'] = thermo.predict_push(group['kinase_conc_infer'].values, group['substrate_conc_infer'].values, *p)

                elif df_dataset_key.loc[exp_name ,'model'] == 'pushpull':
                    df_copy.loc[group.index, 'phospho_conc_predict'] = thermo.predict_pushpull(group['kinase_conc_infer'].values, group['pptase_conc_infer'].values, group['substrate_conc_infer'].values, *p)




                df_copy.loc[group.index, 'phospho_GFP_predict'] = df_copy.loc[group.index, 'phospho_conc_predict'] + phospho_GFP_cutoff


                MSE = np.mean((np.log10(df_copy.loc[group.index, 'phospho_GFP_predict'])-np.log10(df_copy.loc[group.index, 'phospho_GFP_infer']))**2)

                loss += MSE 
                
            return loss
        
        
        def callback(params, weights, errors):
            print("#############################################################")
#             print("Tols:", tols)
#             print("Replicas:", params)
            print("Weights:", weights)
            print("Errors:", errors)
         
            print("Total Loss:", np.mean(errors), "+/-", np.var(errors))
            
            for p in param_dict:
                print(p, np.mean(params[:, param_dict[p]]), "+/-", np.var(params[:, param_dict[p]]))
                         
            end = time.time()

            print("Total Time Elapsed", (end-start)/60, "minutes")
            
            print("#############################################################")
            
            ncols = 4
            nrows = int(np.ceil(len(param_dict)/ncols))
            
            fig = plt.figure(figsize=(4*ncols, 3*nrows))
                        
            for p in param_dict:
                
                i = param_dict[p]
                
                ax = fig.add_subplot(nrows, ncols, i+1)
                
                ax.set_xlabel(p)
                
#                 (l, u) = bounds[i]
                l = np.min([np.min(params[:, i]), x0[i]-4*scale[i]])
                u = np.max([np.max(params[:, i]), x0[i]+4*scale[i]])
        
                t = np.linspace(l, u, 100)
                ax.plot(t, np.exp(-0.5*(t-x0[i])**2/scale[i]**2) / np.sqrt(2*np.pi*scale[i]**2), 'k--')
                                
                sns.histplot(x=params[:, i], weights=weights, binrange=(l, u), ax=ax, stat='density', bins=max(10, N_replicas//10))
                
                
#                 ax.vlines(x0[i])
                
                ax.set_xlim(l, u)

                
            plt.tight_layout()
            plt.show()
            
            
        f0 = loss(x0)
        
        if start_iter == 0:
            params = np.zeros([N_replicas, len(param_dict)], float)
            weights = np.zeros(N_replicas, float)
            
        errors = np.zeros(N_replicas, float)
        
        start = time.time()
        
        for t in range(start_iter, N_iters):
            
            rand.seed(t)

            print("iter:", t)

            candidate_params = np.zeros([N_replicas, len(param_dict)], float)
            candidate_weights = np.zeros(N_replicas, float)
            candidate_errors = np.zeros(N_replicas, float)
            for i in range(N_replicas):

                if i % 10 == 0:
                    print("replica:", i)
                attempt = 0
                while True:

    #                     print("hi", attempt)

                    if t == 0:
                        candidate_params[i] = rand.normal(loc=x0, scale=scale)
    #                         candidate_params[i] = x0
    #                         j = rand.randint(0, len(param_dict))
    #                         candidate_params[i, j] = rand.normal(loc=candidate_params[i, j], scale=scale[j])

                    else:
                        candidate_params[i] = params[rand.choice(np.arange(N_replicas), p=weights)]

                        candidate_params[i] = rand.normal(loc=candidate_params[i], scale=scale)

    #                         j = rand.randint(0, len(param_dict))
    #                         candidate_params[i, j] = rand.normal(loc=candidate_params[i, j], scale=scale[j])


    #                     if (candidate_params[i] > upper).any() or (candidate_params[i] < lower).any():
    #                         print(upper)
    #                         print(lower)
    #                         print(candidate_params[i])
    #                         continue


                    candidate_errors[i] = loss(candidate_params[i])

    #                     print(x0)
    #                     print(candidate_params[i])
    #                     print(attempt, "/", max_attempts, candidate_errors[i], f0, (1.0+tol)*f0)

                    # if within error tolerance, accept and continue
                    if candidate_errors[i] < (1+tol)*f0:
    #                         print(i, attempt, candidate_errors[i])
    #                         print(candidate_params[i])
                        break

                    attempt += 1
                    if attempt >= max_attempts:

                        print("Exceeded max attempts")
                        return params, weights

                if t == 0:
                    candidate_weights[i] = 1.0
                else:
                    S = 0.0
                    for j in range(N_replicas):
                        S += weights[j]*kernel(params[j], candidate_params[i], scale)

                    candidate_weights[i] = prob(candidate_params[i], x0, scale) / S

            candidate_weights /= np.sum(candidate_weights)

            params = candidate_params
            weights = candidate_weights
            errors = candidate_errors

            callback(params, weights, errors)


            yield t, params, weights

    
    param_dict, model_params, x0, bounds = setup_model_params(df_dataset_key)


    print(param_dict)
    print(model_params)
    print(bounds)
    
    
    if df_param_dist is None:
    
        x0 = np.zeros(len(param_dict))

        for exp_name, row in df_dataset_key.iterrows():

            model = row['model']

            params = np.zeros(len(model_params[exp_name]))

            if model == 'substrate_only':
                params[0] = df_dataset_key.loc[exp_name, 'bg_phospho_rate']
            elif model == 'push':
                params[0] = df_dataset_key.loc[exp_name, 'bg_phospho_rate']
                params[1] = df_dataset_key.loc[exp_name, 'kinase_phospho_rate']
                params[2] = df_dataset_key.loc[exp_name, 'kinase_binding_affinity']
            elif model == 'pushpull':
                params[0] = df_dataset_key.loc[exp_name, 'bg_phospho_rate']
                params[1] = df_dataset_key.loc[exp_name, 'kinase_phospho_rate']
                params[2] = df_dataset_key.loc[exp_name, 'kinase_binding_affinity']
                params[3] = df_dataset_key.loc[exp_name, 'pptase_dephospho_rate']
                params[4] = df_dataset_key.loc[exp_name, 'pptase_binding_affinity']

            x0[model_params[exp_name]] = np.log10(params)
            

        columns=['iter', 'replica', 'weight'] + len(param_dict)*['none']
        for p in param_dict:
            columns[3+param_dict[p]] = p

        df_param_dist = pd.DataFrame([], columns=columns)

        df_param_dist.set_index(['iter', 'replica'], inplace=True)
        
        df_param_dist.loc[(-1, 0), :] = [1.0] + list(x0)
                            
        display(df_param_dist)
        
        start_iter = 0
        params = None
        weights = None
    
    else:
        
        x0 = df_param_dist.query("iter==-1 and replica==0").to_numpy().flatten()[1:]
        
        print(x0)
        
        start_iter = df_param_dist.index.get_level_values('iter').max()
        
        data = df_param_dist.query("iter==@start_iter").to_numpy()
        
        start_iter += 1

        params = data[:, 1:]
        
        weights = data[:, 0].flatten()
        

    for (t, params, weights) in solve(df_dataset_key, df_data, model_params, param_dict, bounds, N_replicas=N_replicas, N_iters=N_iters, tol=tol, x0=x0, start_iter=start_iter, params=params, weights=weights):

        data = np.insert(params, 0, weights, axis=1)
        
        df_new = pd.DataFrame(data, columns=df_param_dist.columns)
        df_new['iter'] = t
        df_new['replica'] = np.arange(N_replicas)

        df_new.set_index(['iter', 'replica'], inplace=True)
        
        
        df_param_dist = pd.concat([df_param_dist, df_new])
        
        
        display(df_param_dist)
        
    return df_param_dist
