from IPython.display import display, Markdown

import sys
sys.path.insert(0, '../py_scripts')

import time
import numpy as np
import scipy as sp
import pandas as pd
import scipy.optimize as opt

import push_pull as pp
import noise_models as noise






def fit_push(df_info, df, empty_phospho_noise):


    def solve(df_data, df_info, param_dict, x0, bounds, verbose=False):

        df_copy = df.dropna().copy()

        if verbose:
            start = time.time()

    #     loss_dict = {}
        def func(x):

            loss = 0.0



            for index, row in df_info.iterrows():
                dataset = row['dataset']

                model_params = 10**np.array(x)[param_dict[dataset]]

                df_data = df_copy.query("dataset==@dataset")

    #            
                if row['model'] == 'push':


                    df_copy.loc[df_data.index, 'SpT_conc_predict'] = pp.PushAmp().predict_all(df_data[['WT_conc_infer', 'ST_conc_infer']].values, model_params)[:, 0]


                df_copy.loc[df_data.index, 'SpT_GFP_predict'] = df_copy.loc[df_data.index, 'SpT_conc_predict'] + np.median(empty_phospho_noise.get_GFP())

    #             loss_dict[dataset] = np.mean((np.log10(df_copy.loc[df_data.index, 'SpT_GFP_predict'])-np.log10(df_copy.loc[df_data.index, 'SpT_GFP_infer']))**2)

                prob_empty = df_copy.loc[df_data.index, 'total_prob_empty']  

                loss += np.mean((1-prob_empty)*(np.log10(df_copy.loc[df_data.index, 'SpT_GFP_predict'])-np.log10(df_copy.loc[df_data.index, 'SpT_GFP_infer']))**2)


            # add small tether regularization to initial conditions
            loss += 1e-6*np.sum((x-np.array(x0))**2)

            return loss




        print("Initial Loss:", func(x0))

        def callback(x):
            print(func(x), 1e-6*np.sum((x-np.array(x0))**2), x)
    #         print(loss_dict)

    #     res = opt.minimize(func, x0, method='L-BFGS-B', 
    #                        jac='2-point', bounds=bounds, 
    #                        options={'iprint':101, 'eps': 1e-6, 
    #                                 'gtol': 1e-6, 'ftol':1e-6},
    #                       callback=callback)
        res = opt.minimize(func, x0, method='BFGS', 
                           jac='2-point', 
                           options={'eps': 1e-6, 'gtol': 1e-4, 'disp': True},
                          callback=callback)


        print("Final Loss:", res.fun, func(res.x))

        end = time.time()

        print("Time Elapsed", end-start, "seconds")

        print(res)


        return res



    zippers = {}
    kinases = {}

    param_dict = {}

    param_labels = [r"$\log_{10}(v_{bg}^p)$"]
    x0 = [-2.0]
    bounds = [(None, None)]


    param_index = 1
    for index, row in df_info.iterrows():

        dataset = row['dataset']
        zipper = row['zipper']
        kinase = row['kinase']
        model = row['model']

        if model == 'push':

            # assign background phospho rate
            param_dict[dataset] = [0]

            # assign kinase phospho rate
            if kinase not in kinases:
                kinases[kinase] = param_index
                param_labels.append(kinase + ": " + r"$\log_{10}(v_{WS}^p)$")
                x0.append(-1.0)
                bounds.append((None, None))
                param_index += 1

            param_dict[dataset].append(kinases[kinase])

            # assign zipper binding affinity
            if zipper not in zippers:
                zippers[zipper] = param_index
                param_labels.append(zipper + ": " + r"$\log_{10}(\alpha_{WS})$")
                x0.append(3.0)
                bounds.append((None, None))
                param_index += 1

            param_dict[dataset].append(zippers[zipper])


    print(param_labels)
    print(param_dict)

    print(x0)
    print(bounds)

    res = solve(df, df_info, param_dict, x0, bounds, verbose=True)

    
    for index, row in df_info.iterrows():
    
        dataset = row['dataset']
        model_params = 10**res.x[param_dict[dataset]]
        df_info.loc[index, 'bg_phospho_rate'] = model_params[0]
        df_info.loc[index, 'kinase_phospho_rate'] = model_params[1]
        df_info.loc[index, 'kinase_bind_affin'] = model_params[2]
        
        
    display(df_info)
    
    return res, param_dict, param_labels