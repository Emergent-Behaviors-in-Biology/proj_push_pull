from IPython.display import display, Markdown

import sys
sys.path.insert(0, '../py_scripts')

import time
import numpy as np
import scipy as sp
import pandas as pd
import scipy.optimize as opt

# import push_pull as pp
import noise_models as noise
import thermo_models as thermo






def fit_push(df_dataset_key, df_data, phospho_GFP_cutoff):


    def solve(df_dataset_key, df_data, model_params, param_dict, x0, bounds, verbose=False):

        df_copy = df_data.dropna().copy()

        if verbose:
            start = time.time()

    #     loss_dict = {}
        def func(x):

            loss = 0.0



            for exp_name, row in df_dataset_key.iterrows():

                
                params = 10**np.array(x)[model_params[exp_name]]

                df_tmp = df_copy.query("exp_name==@exp_name")

                if row['model'] == 'substrate_only':
                    
                    df_copy.loc[df_tmp.index, 'SpT_conc_predict'] = thermo.predict_substrate_only(df_tmp['ST_conc_infer'].values, *params)
                                        
                elif row['model'] == 'non-pplatable':
                    
                    df_copy.loc[df_tmp.index, 'SpT_conc_predict'] = thermo.predict_nonpplatable(df_tmp['ST_conc_infer'].values)
                    
                elif row['model'] == 'push':

                    df_copy.loc[df_tmp.index, 'SpT_conc_predict'] = thermo.predict_push(df_tmp['WT_conc_infer'].values, df_tmp['ST_conc_infer'].values, *params)

                elif row['model'] == 'pushpull':
                    df_copy.loc[df_tmp.index, 'SpT_conc_predict'] = thermo.predict_pushpull(df_tmp['WT_conc_infer'].values, df_tmp['ET_conc_infer'].values, df_tmp['ST_conc_infer'].values, *params)

                    
                

                df_copy.loc[df_tmp.index, 'SpT_GFP_predict'] = df_copy.loc[df_tmp.index, 'SpT_conc_predict'] + phospho_GFP_cutoff

                
                MSE = np.mean((np.log10(df_copy.loc[df_tmp.index, 'SpT_GFP_predict'])-np.log10(df_copy.loc[df_tmp.index, 'SpT_GFP_infer']))**2)
                var = np.mean((np.log10(df_copy.loc[df_tmp.index, 'SpT_GFP_infer'])-np.log10(df_copy.loc[df_tmp.index, 'SpT_GFP_infer']).mean())**2)
                
                loss += MSE 
                
                
            # add small tether regularization to initial conditions
            loss += 1e-6*np.sum((x-np.array(x0))**2)

            return loss




        

        def callback(x):
            print("#############################################################")
            print("Total Loss:", func(x), "Regularization:", 1e-6*np.sum((x-np.array(x0))**2))
            
            for p in param_dict:
                print(p, x0[param_dict[p]], x[param_dict[p]])
                         
            end = time.time()

            print("Total Time Elapsed", (end-start)/60, "minutes")

            
        callback(x0)
            

    #     res = opt.minimize(func, x0, method='L-BFGS-B', 
    #                        jac='2-point', bounds=bounds, 
    #                        options={'iprint':101, 'eps': 1e-6, 
    #                                 'gtol': 1e-6, 'ftol':1e-6},
    #                       callback=callback)
        res = opt.minimize(func, x0, method='BFGS', 
                           jac='2-point', 
                           options={'eps': 1e-6, 'gtol': 1e-4, 'disp': True},
                          callback=callback)


        callback(res.x)

        print(res)


        return res


    # map of parameters names to indices
    param_dict = {'bg_phospho_rate': 0}
    # map of datasets to lists of relevant parameters
    model_params = {}

    # list of initial conditions
    x0 = [-2.0]
    # list of parameter bounds
    bounds = [(None, None)]


    param_index = 1
    for exp_name, row in df_dataset_key.iterrows():

        model = row['model']
        
        if model == 'substrate_only':
            # assign background phospho rate
            model_params[exp_name] = [param_dict['bg_phospho_rate']]
            
        elif model == 'non-pplatable':
            # no parameters
            model_params[exp_name] = []

        elif model == 'push':

            # assign background phospho rate
            model_params[exp_name] = [param_dict['bg_phospho_rate']]

            # assign kinase phospho rate
            kinase = row['kinase_variant']
            if kinase not in param_dict:
                param_dict[kinase] = len(param_dict)
                x0.append(-1.0)
                bounds.append((None, None))

            model_params[exp_name].append(param_dict[kinase])

            # assign kinase zipper binding affinity
            zipper = row['kinase_zipper']
            if zipper not in param_dict:
                param_dict[zipper] = len(param_dict)
                x0.append(3.0)
                bounds.append((None, None))

            model_params[exp_name].append(param_dict[zipper])
            
        elif model == 'pushpull':

            # assign background phospho rate
            model_params[exp_name] = [param_dict['bg_phospho_rate']]

            # assign kinase phospho rate
            kinase = row['kinase_variant']
            if kinase not in param_dict:
                param_dict[kinase] = len(param_dict)
                x0.append(-1.0)
                bounds.append((None, None))

            model_params[exp_name].append(param_dict[kinase])

            # assign kinase zipper binding affinity
            zipper = row['kinase_zipper']
            if zipper not in param_dict:
                param_dict[zipper] = len(param_dict)
                x0.append(3.0)
                bounds.append((None, None))

            model_params[exp_name].append(param_dict[zipper])
            
            # assign pptase phospho rate
            pptase = row['pptase_variant']
            if pptase not in param_dict:
                param_dict[pptase] = len(param_dict)
                x0.append(-1.0)
                bounds.append((None, None))

            model_params[exp_name].append(param_dict[pptase])

            # assign pptase zipper binding affinity
            zipper = row['pptase_zipper']
            if zipper not in param_dict:
                param_dict[zipper] = len(param_dict)
                x0.append(3.0)
                bounds.append((None, None))

            model_params[exp_name].append(param_dict[zipper])


    print(param_dict)
    print(model_params)

    print(x0)
    print(bounds)

    res = solve(df_dataset_key, df_data, model_params, param_dict, x0, bounds, verbose=True)

    
    for exp_name, row in df_dataset_key.iterrows():
    
        model = row['model']
        
        params = 10**res.x[model_params[exp_name]]
        
        if model == 'substrate_only':
            df_dataset_key.loc[exp_name, 'bg_phospho_rate'] = params[0]
        elif model == 'push':
            df_dataset_key.loc[exp_name, 'bg_phospho_rate'] = params[0]
            df_dataset_key.loc[exp_name, 'kinase_phospho_rate'] = params[1]
            df_dataset_key.loc[exp_name, 'kinase_binding_affinity'] = params[2]
        elif model == 'pushpull':
            df_dataset_key.loc[exp_name, 'bg_phospho_rate'] = params[0]
            df_dataset_key.loc[exp_name, 'kinase_phospho_rate'] = params[1]
            df_dataset_key.loc[exp_name, 'kinase_binding_affinity'] = params[2]
            df_dataset_key.loc[exp_name, 'pptase_dephospho_rate'] = params[3]
            df_dataset_key.loc[exp_name, 'pptase_binding_affinity'] = params[4]
              
        
        
    display(df_dataset_key)
    
    return res, param_dict
