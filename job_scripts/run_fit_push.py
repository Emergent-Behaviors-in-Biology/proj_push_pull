import sys, os
sys.path.insert(0, '../py_scripts/')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

from IPython.display import display, Markdown

import numpy as np
import scipy as sp
import pandas as pd

import numpy.random as rand

import pickle


import time
import pickle


import noise_models as noise
import model_fitting as fit
import fig_plot as fplot
import thermo_models as thermo


seed = 42

print("Seed:", seed)

rand.seed(seed)


# name of dataset folder
# label = "21_10_15_medhighgating"
label = "22_05_05_twolayer"

components = ["phospho", "substrate", "kinase", 'pptase', 'kinase2', 'kinase2phospho']


df_dataset_key = pd.read_csv("../data/"+label+"/dataset_key.csv", sep='\s*,\s*', engine='python').set_index("exp_name")
display(df_dataset_key)

df_MOCU_key = pd.read_csv("../data/"+label+"/MOCU_key.csv", sep='\s*,\s*', engine='python').set_index("component")
display(df_MOCU_key)


# load datasets

df_list = []
for exp_name, row in df_dataset_key.iterrows():
    
    df = pd.read_csv("../data/{}/{}.csv".format(label, row['file_name']))
    df = df.drop("Unnamed: 0", axis=1, errors='ignore').reset_index(drop=True)

    df = df.rename(columns={row['substrate_col']:'substrate_anti_exp', 
                         row['phospho_col']:'phospho_anti_exp', 
                         row['kinase_col']:'kinase_anti_exp'})
    
    if row['model'] == 'pushpull' or row['model'] == 'two_layer':
        df = df.rename(columns={row['pptase_col']:'pptase_anti_exp'})
    else:
        df['pptase_anti_exp'] = 1e-8
        
    
    if row['model'] == 'two_layer':
        df = df.rename(columns={row['kinase2_col']:'kinase2_anti_exp'})
        df = df.rename(columns={row['kinase2phospho_col']:'kinase2phospho_anti_exp'})
        df['kinase2phospho_anti_exp'] = 1/3 * df['kinase2phospho_anti_exp']
    else:
        df['kinase2_anti_exp'] = 1e-8
        df['kinase2phospho_anti_exp'] = 1e-8
        
   
    
    df['exp_name'] = exp_name
    df.index.rename('cell_index', inplace=True)
    df_list.append(df)
    
# dataframe containing all datasets   
df_data = pd.concat(df_list) #.drop("Unnamed: 0", axis=1, errors='ignore')
df_data = df_data.reset_index().set_index(['cell_index', 'exp_name'])
df_data = df_data.reorder_levels(df_data.index.names[::-1])

print(len(df_data.index))
df_data.dropna(inplace=True)
print(len(df_data.index))
df_data = df_data[(df_data[df_data.columns] > 0.0).all(axis=1)]
print(len(df_data.index))


# setup noise model dictionary
noise_models = {c:dict() for c in components}
print(noise_models)


display(df_data)







# points per bin
ppbin = 100

for c in components:
    
    # distribution of antibody values w GFP and GFP for non-empty cells
    df = pd.read_csv("../data/{}/{}.csv".format(label, df_MOCU_key.loc[c, 'file_name']))    
    anti = df[df_MOCU_key.loc[c, 'anti_col_name']].values
    GFP = df[df_MOCU_key.loc[c, 'GFP_col_name']].values
    idx = (anti > 0.0) & (GFP > 0.0)
    noise_models[c]['anti'] = noise.Density(anti[idx], ppbin=ppbin)
    noise_models[c]['GFP'] = noise.Density(GFP[idx], ppbin=ppbin)
    
    # distribution of antibodies and GFP for empty cells
    df = pd.read_csv("../data/{}/{}.csv".format(label, df_MOCU_key.loc['empty_'+c, 'file_name']))
    anti = df[df_MOCU_key.loc['empty_'+c, 'anti_col_name']].values
    GFP = df[df_MOCU_key.loc['empty_'+c, 'GFP_col_name']].values
    idx = (anti > 0.0) & (GFP > 0.0)
    noise_models[c]['empty_anti'] = noise.Density(anti[idx], ppbin=ppbin)
    noise_models[c]['empty_GFP'] = noise.Density(GFP[idx], ppbin=ppbin)
    
    
    
# points per bin
ppbin = 10

gaussian_cutoff_percentile = 0.99
empty_prior = 0.5

for c in components:
        
    noise_models[c]['empty_anti2GFP'] =  noise.RandomConditionalNoise(noise_models[c]['empty_anti'].get_data(), 
                                            noise_models[c]['empty_GFP'].get_data(), ppbin=ppbin)

    noise_models[c]['nonempty_anti2GFP'] =  noise.RandomConditionalNoise(noise_models[c]['anti'].get_data(), 
                                            noise_models[c]['GFP'].get_data(), ppbin=ppbin)
        
        
    noise_models[c]['composite_anti2GFP'] = noise.CompositeConditionalNoise(noise_models[c]['empty_anti2GFP'], 
                                                                            noise_models[c]['nonempty_anti2GFP'],
                                                                           empty_prob=empty_prior, 
                                                                            cutoff_percent=gaussian_cutoff_percentile)
    

# reverse noise model for pplated substrate

c = 'phospho'

noise_models[c]['empty_GFP2anti'] =  noise.RandomConditionalNoise(noise_models[c]['empty_GFP'].get_data(), 
                                            noise_models[c]['empty_anti'].get_data(), ppbin=ppbin)

noise_models[c]['nonempty_GFP2anti'] =  noise.RandomConditionalNoise(noise_models[c]['GFP'].get_data(), 
                                        noise_models[c]['anti'].get_data(), ppbin=ppbin)

noise_models[c]['composite_GFP2anti'] = noise.CompositeConditionalNoise(noise_models[c]['empty_GFP2anti'], 
                                                                        noise_models[c]['nonempty_GFP2anti'],
                                                                       empty_prob=empty_prior, 
                                                                        cutoff_percent=gaussian_cutoff_percentile)



# reverse noise model for pplated substrate

c = 'kinase2phospho'

noise_models[c]['empty_GFP2anti'] =  noise.RandomConditionalNoise(noise_models[c]['empty_GFP'].get_data(), 
                                            noise_models[c]['empty_anti'].get_data(), ppbin=ppbin)

noise_models[c]['nonempty_GFP2anti'] =  noise.RandomConditionalNoise(noise_models[c]['GFP'].get_data(), 
                                        noise_models[c]['anti'].get_data(), ppbin=ppbin)

noise_models[c]['composite_GFP2anti'] = noise.CompositeConditionalNoise(noise_models[c]['empty_GFP2anti'], 
                                                                        noise_models[c]['nonempty_GFP2anti'],
                                                                       empty_prob=empty_prior, 
                                                                        cutoff_percent=gaussian_cutoff_percentile)


for c in components:
    df_data[c+'_GFP_infer'] = 0.0
    
for exp_name, row in df_dataset_key.iterrows():
    
    print(exp_name)
        
    df_tmp = df_data.query("exp_name==@exp_name").dropna()
    
    for c in components:
        
        # a weird way to check for nans or empty values
        if row[c+'_col'] != row[c+'_col']:
            continue
            
        anti = df_data.loc[df_tmp.index, c+'_anti_exp']
    
        df_data.loc[df_tmp.index, c+'_GFP_infer'] = noise_models[c]['composite_anti2GFP'].transform(anti)
        


display(df_data)



try:
    df_params = pd.read_csv("../data/"+label+"/model_params.csv", sep=',', engine='python')   
except:
    df_params = None

# Uncomment this to overwrite all previous fits
# df_params = None
    
display(df_params)



df_params = fit.fit(df_dataset_key.query("model=='two_layer'"), df_data, df_params=df_params)

display(df_params)

df_params.to_csv("../data/"+label+"/model_params.csv", sep=',', index=False)



df_params = fit.calc_error(df_dataset_key.query("model=='two_layer'"), df_data, df_params=df_params)

display(df_params)

df_params.to_csv("../data/"+label+"/model_params.csv", sep=',', index=False)


prefit_params, param_to_index, dataset_to_params, x0, bounds = fit.setup_model_params(df_dataset_key, df_params=df_params)

x = np.zeros_like(x0)
for p in param_to_index:
    x[param_to_index[p]] = df_params.query("name==@p").iloc[0]['val']

args = (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params)

fit.predict(x, args, df_data)

display(df_data)

print(len(df_data))
print(len(df_data.dropna()))




df_data['phospho_anti_predict'] = 0.0
df_data['kinase2phospho_anti_predict'] = 0.0

for exp_name, row in df_dataset_key.iterrows():
    
    df_tmp = df_data.query("exp_name==@exp_name").dropna()

    df_data.loc[df_tmp.index, 'phospho_anti_predict'] = noise_models['phospho']['composite_GFP2anti'].transform(df_data.loc[df_tmp.index, 'phospho_GFP_predict'])

    df_data.loc[df_tmp.index, 'kinase2phospho_anti_predict'] = noise_models['kinase2phospho']['composite_GFP2anti'].transform(df_data.loc[df_tmp.index, 'kinase2phospho_GFP_predict'])

    
display(df_data)


df_data.to_csv("../data/"+label+"/model_predictions.csv", sep=',')