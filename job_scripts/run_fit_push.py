import sys, os
sys.path.insert(0, '../py_scripts/')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

from IPython.display import display, Markdown

import numpy as np
import scipy as sp
import pandas as pd

import numpy.random as rand

import pickle

import noise_models as noise
import model_fitting as fit




seed = int(sys.argv[1])

print("Seed:", seed)

rand.seed(seed)




# list of of datasets to load
# dataset, zipper variant, kinase variant, model
s_list = [
    ['E+E', 'E+E', 'wt', 'push'],
    ['I+E', 'I+E', 'wt', 'push'],
    ['RR+A', 'RR+A', 'wt', 'push'],
    ['S+A', 'S+A', 'wt', 'push'],
    ['S+E', 'S+E', 'wt', 'push'],
    ['wt', 'generic', 'wt', 'push'],
    ['KD', 'generic', 'KD', 'push'],
    ['R460A', 'generic', 'R460A', 'push'],
    ['R460K', 'generic', 'R460K', 'push'],
    ['R460S', 'generic', 'R460S', 'push'],
         ]
# dataframe containing info for each data set
df_info = pd.DataFrame(s_list, columns=['dataset', 'zipper', 'kinase', 'model'])
df_info['seed'] = seed

display(df_info)

# load datasets

df_list = []
for index, row in df_info.iterrows():
    df = pd.read_csv("../data/push_data/{}.csv".format(row['dataset'])) 
    
    df = df.drop("Unnamed: 0", axis=1, errors='ignore').sample(frac=1.0, replace=True, random_state=seed).reset_index(drop=True)
        
    df['dataset'] = row['dataset']      
    df_list.append(df)
    
# dataframe containing all datasets   
df_data = pd.concat(df_list).drop("Unnamed: 0", axis=1, errors='ignore')
df_data.set_index("dataset", inplace=True, append=True)
df_data = df_data.reorder_levels(df_data.index.names[::-1])
df_data = df_data.rename(columns={'WT_anti': 'WT_anti_exp', 'ST_anti': 'ST_anti_exp', 'SpT_anti': 'SpT_anti_exp'})



# # print(len(df.index))
# # df = df[(df[df.columns[:-1]] > 0).all(axis=1)].rename(columns={'WT_anti': 'WT_anti_exp', 'ST_anti': 'ST_anti_exp', 'SpT_anti': 'SpT_anti_exp'})
# # print(len(df.index))

# record fraction of phospho substrate
df_data['Sp_frac_anti_exp'] = df_data['SpT_anti_exp'] / df_data['ST_anti_exp']


display(df_data)



# load noise models


writer_noise = noise.Anti2GFPNoise("../data/noise_data/Flag noise.csv", 
                                   'HA', 'GFP', ppbin=10, verbose=False)

empty_writer_noise = noise.Anti2GFPNoise("../data/noise_data/Empty Cell.csv", 
                                   'HA', 'GFP', ppbin=10, verbose=False)


    
substrate_noise = noise.Anti2GFPNoise("../data/noise_data/Phopho_Myc noise.csv", 
                                   'Myc', 'GFP', ppbin=10, verbose=False)

empty_substrate_noise = noise.Anti2GFPNoise("../data/noise_data/Empty Cell.csv", 
                                   'Myc', 'GFP', ppbin=10, verbose=False)



    
phospho_noise = noise.Anti2GFPNoise("../data/noise_data/Phopho_Myc noise.csv", 
                                   'Phospho', 'GFP', ppbin=10, verbose=False)

empty_phospho_noise = noise.Anti2GFPNoise("../data/noise_data/Empty Cell.csv", 
                                   'Phospho', 'GFP', ppbin=10, verbose=False)


combined_phospho_noise = noise.Anti2GFPNoise("../data/noise_data/Phopho_Myc noise.csv", 
                                   'Phospho', 'GFP', ppbin=10, verbose=False)
combined_phospho_noise.add_cells(empty_phospho_noise)



# inverse_phospho_noise = noise.GFP2AntiNoise("../data/noise_data/Phopho_Myc noise.csv", 
#                                    'GFP', 'Phospho', ppbin=10, verbose=False)

# inverse_empty_phospho_noise = noise.GFP2AntiNoise("../data/noise_data/Empty Cell.csv", 
#                                    'GFP', 'Phospho', ppbin=10, verbose=False)

# inverse_phospho_noise.add_cells(inverse_empty_phospho_noise)




# fit empty cell noise models and rescale antibody data

for index, row in df_info.iterrows():
    
    dataset = row['dataset']
    
    df_tmp = df_data.query("dataset==@dataset")
    
    ################################################################
    
    
    (writer_empty_frac, writer_anti_scale) = noise.calc_mixture(df_tmp['WT_anti_exp'], 
                                                                empty_writer_noise, writer_noise, seed=seed, maxiter=10000)

    
    df_info.loc[index, 'WT_empty_frac'] = writer_empty_frac
    df_info.loc[index, 'WT_anti_scale'] = writer_anti_scale
    
        
    df_data.loc[df_tmp.index, 'WT_anti_rescaled'] = df_tmp['WT_anti_exp'] / 10**writer_anti_scale
        
        
    df_data.loc[df_tmp.index, 'WT_prob_empty'] = noise.calc_prob_empty(df_data.loc[df_tmp.index, 'WT_anti_rescaled'], 
                                                                 writer_empty_frac, 
                                                                 empty_writer_noise, writer_noise, maxiter=10000)
    
    
    ################################################################
    
    (substrate_empty_frac, substrate_anti_scale) = noise.calc_mixture(df_tmp['ST_anti_exp'], 
                                                                empty_substrate_noise, substrate_noise, seed=seed)

        
    df_info.loc[index, 'ST_empty_frac'] = substrate_empty_frac
    df_info.loc[index, 'ST_anti_scale'] = substrate_anti_scale
        
            
    df_data.loc[df_tmp.index, 'ST_anti_rescaled'] = df_tmp['ST_anti_exp'] / 10**substrate_anti_scale
    
        
    df_data.loc[df_tmp.index, 'ST_prob_empty'] = noise.calc_prob_empty(df_data.loc[df_tmp.index, 'ST_anti_rescaled'], 
                                                                 substrate_empty_frac, 
                                                                 empty_substrate_noise, substrate_noise)
    

    
display(df_info)



# convert antibody values to inferred GFP values using noise models

zero = 0.0

for index, row in df_info.iterrows():
    
    dataset = row['dataset']
    
    df_tmp = df_data.query("dataset==@dataset")
    
    # convert antibody measurements to GFP measurements
    GFP_infer, anti_bin, GFP_bin = writer_noise.anti_to_GFP(df_data.loc[df_tmp.index, 'WT_anti_rescaled'])
    df_data.loc[df_tmp.index, 'WT_GFP_infer'] = GFP_infer
#     df_data.loc[df_tmp.index, 'WT_anti_bin'] = anti_bin
#     df_data.loc[df_tmp.index, 'WT_GFP_bin'] = GFP_bin
    df_data.loc[df_tmp.index, 'WT_conc_infer'] = np.maximum(df_data.loc[df_tmp.index, 'WT_GFP_infer'] - np.median(empty_writer_noise.get_GFP()), zero)

    GFP_infer, anti_bin, GFP_bin = substrate_noise.anti_to_GFP(df_data.loc[df_tmp.index, 'ST_anti_rescaled'])
    df_data.loc[df_tmp.index, 'ST_GFP_infer'] = GFP_infer
#     df_data.loc[df_tmp.index, 'ST_anti_bin'] = anti_bin
#     df_data.loc[df_tmp.index, 'ST_GFP_bin'] = GFP_bin
    df_data.loc[df_tmp.index, 'ST_conc_infer'] = np.maximum(df_data.loc[df_tmp.index, 'ST_GFP_infer'] - np.median(empty_substrate_noise.get_GFP()), zero)

    
    GFP_infer, anti_bin, GFP_bin = combined_phospho_noise.anti_to_GFP(df_data.loc[df_tmp.index, 'SpT_anti_exp'])
    df_data.loc[df_tmp.index, 'SpT_GFP_infer'] = GFP_infer
#     df_data.loc[df_tmp.index, 'SpT_anti_bin'] = anti_bin
#     df_data.loc[df_tmp.index, 'SpT_GFP_bin'] = GFP_bin
    df_data.loc[df_tmp.index, 'SpT_conc_infer'] = np.maximum(df_data.loc[df_tmp.index, 'SpT_GFP_infer'] - np.median(empty_phospho_noise.get_GFP()), zero)

    df_data.loc[df_tmp.index, 'Sp_frac_GFP_infer'] = df_data.loc[df_tmp.index, 'SpT_GFP_infer'] / df_data.loc[df_tmp.index, 'ST_GFP_infer']
    df_data.loc[df_tmp.index, 'Sp_frac_conc_infer'] = df_data.loc[df_tmp.index, 'SpT_conc_infer'] / df_data.loc[df_tmp.index, 'ST_conc_infer']
    
    
    df_data.loc[df_tmp.index, 'total_prob_empty'] = 1.0  - (1-df_data.loc[df_tmp.index, 'ST_prob_empty'])*(1-df_data.loc[df_tmp.index, 'WT_prob_empty'])
    
    
    
display(df_data)
print(len(df_data))
print(len(df_data.dropna()))



# fit the thermodynamic model

res, param_dict, param_labels = fit.fit_push(df_info, df_data, empty_phospho_noise)



with open('/projectnb/biophys/jrocks/proj_push_pull/data/push_data_{0:08d}.pkl'.format(seed), 'wb') as pkl_file:

    data = {'df_info': df_info}
    pickle.dump(data, pkl_file)