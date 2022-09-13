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
label = '220810_secondlayer'

components = ["phospho", "substrate", "kinase", 'pptase', 'kinase2', 'kinase2_phospho']


df_dataset_key = pd.read_csv("../data/"+label+"/dataset_key.csv", sep='\s*,\s*', engine='python').set_index("exp_name")
display(df_dataset_key)

df_MOCU_key = pd.read_csv("../data/"+label+"/MOCU_key.csv", sep='\s*,\s*', engine='python').set_index("component")
display(df_MOCU_key)


# load datasets

df_list = []
for exp_name, row in df_dataset_key.iterrows():
    
    df = pd.read_csv("../data/{}/{}.csv".format(label, row['file_name']))
    df = df.drop("Unnamed: 0", axis=1, errors='ignore').sample(frac=1.0, replace=False, random_state=seed).reset_index(drop=True)
#     df = df.drop("Unnamed: 0", axis=1, errors='ignore').sample(n=200, replace=False, random_state=seed).reset_index(drop=True)
# 
    df = df.rename(columns={row['substrate_col']:'substrate_anti_exp', 
                         row['phospho_col']:'phospho_anti_exp', 
                         row['kinase_col']:'kinase_anti_exp'})
    
    if row['model'] == 'pushpull' or row['model'] == 'two_layer' or row['model'] == 'two_layer_nowriter' or row['model'] == 'two_layer_noeraser':
        df = df.rename(columns={row['pptase_col']:'pptase_anti_exp'})
    else:
        df['pptase_anti_exp'] = 1e-8
        
    
    if row['model'] == 'two_layer' or row['model'] == 'two_layer_nowriter' or row['model'] == 'two_layer_noeraser':
        df = df.rename(columns={row['kinase2_col']:'kinase2_anti_exp'})
        df = df.rename(columns={row['kinase2_phospho_col']:'kinase2_phospho_anti_exp'})
        df['kinase2_phospho_anti_exp'] = df['kinase2_phospho_anti_exp']

    else:
        df['kinase2_anti_exp'] = 1e-8
        df['kinase2_phospho_anti_exp'] = 1e-8
        
   
    df.drop(df.columns.difference(['substrate_anti_exp','phospho_anti_exp', 'kinase_anti_exp', 'pptase_anti_exp', 'kinase2_anti_exp', 'kinase2_phospho_anti_exp']), 1, inplace=True)
    
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


display(df_data)




# setup noise model dictionary
noise_models = {c:dict() for c in components}
print(noise_models)

try:
    with open("../data/"+label+"/noise_model_params.pkl", 'rb') as pkl_file:
        noise_model_params = pickle.load(pkl_file)
except:
    noise_model_params = {}
    
    
    
display(noise_model_params)



# points per bin
ppbin = 100

for c in components:
    
    # phospho noise is measured using mCherry instead of GFP, so first need to convert to GFP
    if c == 'phospho':
        
        # mCherry to GFP conversion
        df = pd.read_csv("../data/{}/GFP-mCherry.csv".format(label))    
        GFP = df['GFP'].values
        mCherry = df['mCherry'].values
        idx = (mCherry > 0.0) & (GFP > 0.0)
        GFP = GFP[idx]
        mCherry = mCherry[idx]
        noise_models[c]['mCherry2GFP'] = noise.LinearNoise(mCherry, GFP)
        
        
        
    # distribution of antibodies and GFP for non-empty cells
    df = pd.read_csv("../data/{}/{}.csv".format(label, df_MOCU_key.loc[c, 'file_name']))    
    anti = df[df_MOCU_key.loc[c, 'anti_col_name']].values
    GFP = df[df_MOCU_key.loc[c, 'GFP_col_name']].values
    idx = (anti > 0.0) & (GFP > 0.0)
    anti = anti[idx]
    GFP = GFP[idx]
    
    
    
    noise_models[c]['anti'] = noise.BackgroundDist(anti, ppbin=ppbin)
    noise_models[c]['GFP'] = noise.BackgroundDist(GFP, ppbin=ppbin)
    
    # linear mode for converting antibody to GFP measurements
    noise_models[c]['anti2GFP'] = noise.LinearNoise(anti, GFP)
    
    
    # distribution of antibodies and GFP for empty cells
    df = pd.read_csv("../data/{}/{}.csv".format(label, df_MOCU_key.loc['empty_'+c, 'file_name']))
    anti = df[df_MOCU_key.loc['empty_'+c, 'anti_col_name']].values
    GFP = df[df_MOCU_key.loc['empty_'+c, 'GFP_col_name']].values
    idx = (anti > 0.0) & (GFP > 0.0)
    anti = anti[idx]
    GFP = GFP[idx]
    
    # if phospho, convert mCherry to GFP
    if c == 'phospho':
        GFP = noise_models[c]['mCherry2GFP'].transform(GFP)
    
    noise_models[c]['anti_background'] = noise.BackgroundDist(anti, ppbin=ppbin)
    noise_models[c]['GFP_background'] = noise.BackgroundDist(GFP, ppbin=ppbin)

    
    # convert antibody background to GFP units
    empty_anti_as_GFP = noise_models[c]['anti2GFP'].transform(noise_models[c]['anti_background'].get_data())       
    
    noise_models[c]['anti_as_GFP_background'] = noise.BackgroundDist(empty_anti_as_GFP, ppbin=ppbin)
    
    # lognormal noise model with background
    noise_models[c]['GFP2MOCU'] = noise.LogNormalBGNoise(noise_models[c]['anti_as_GFP_background'])
        
    

    
for c in components:
    df_data[c+'_GFP_infer'] = 0.0
    
for exp_name, row in df_dataset_key.iterrows():
    
    print(exp_name)
        
    df_tmp = df_data.query("exp_name==@exp_name")
    
    for c in components:

        # a weird way to check for nans or empty values
        if row[c+'_col'] != row[c+'_col']:
            continue
            
        
        df_data.loc[df_tmp.index, c+'_GFP_infer'] = noise_models[c]['anti2GFP'].transform(df_data.loc[df_tmp.index, c+'_anti_exp'])
        
        
display(df_data)



for exp_name, row in df_dataset_key.iterrows():
    
    print(exp_name)
        
    df_tmp = df_data.query("exp_name==@exp_name")
    
    if exp_name not in noise_model_params:
        noise_model_params[exp_name] = {}
    
    for c in components:
        
        if c == 'phospho' or c == 'kinase2_phospho':
            continue

        # a weird way to check for nans or empty values
        if row[c+'_col'] != row[c+'_col']:
            continue
            
            
        bg_noise = noise_models[c]['anti_as_GFP_background']
        MOCU_noise = noise_models[c]['GFP2MOCU']
        
        GFP = df_data.loc[df_tmp.index, c+'_GFP_infer'] 
        
        # fit background noise model if doesn't exist
        if c not in noise_model_params[exp_name]:

            params = fit.fit_bg_noise(GFP, MOCU_noise) 
  
        
            (mu, sigma) = params.tolist()
            c0 = np.exp(mu)
        
            noise_model_params[exp_name][c] = (c0, sigma)

            
display(noise_model_params)

with open("../data/"+label+"/noise_model_params.pkl", 'wb') as pkl_file:
    pickle.dump(noise_model_params, pkl_file)
    
    
    
    

    
for c in components:
    
    if c == 'phospho' or c == 'kinase2_phospho':
            continue
    
    df_data[c+'_MOCU_infer'] = 0.0

for exp_name, row in df_dataset_key.iterrows():
    
    print(exp_name)
        
    df_tmp = df_data.query("exp_name==@exp_name")
    
    for c in components:
        
        if c == 'phospho' or c == 'kinase2_phospho':
            continue

        # a weird way to check for nans or empty values
        if row[c+'_col'] != row[c+'_col']:
            continue
            
        MOCU_noise = noise_models[c]['GFP2MOCU']
        
        (c0, sigma) = noise_model_params[exp_name][c]
            
        GFP = df_data.loc[df_tmp.index, c+'_GFP_infer'] 
        
        MOCU = MOCU_noise.cal_mean_conc(GFP, c0*np.ones_like(GFP), sigma)
        
        df_data.loc[df_tmp.index, c+'_MOCU_infer']  = MOCU
        
        df_data.loc[df_tmp.index, c+'_GFP_denoise'] = MOCU + noise_models[c]['anti_as_GFP_background'].mean

display(df_data)


    
    

try:
    df_params = pd.read_csv("../data/"+label+"/model_params.csv", sep=',', engine='python')   
except:
    df_params = None

# Uncomment this to overwrite all previous fits
# df_params = None
    
display(df_params)


df_params = fit.fit(df_dataset_key.query("model=='two_layer' or model=='two_layer_nowriter' or model=='two_layer_noeraser'"), df_data, df_params=df_params, noise_models=noise_models)

display(df_params)


df_params.to_csv("../data/"+label+"/model_params.csv", sep=',', index=False)



# df_params = fit.calc_error(df_dataset_key.query("model=='two_layer' or model=='two_layer_nowriter' or model=='two_layer_noeraser'"), df_data, df_params=df_params)

# display(df_params)


# df_params.to_csv("../data/"+label+"/model_params.csv", sep=',', index=False)



prefit_params, param_to_index, dataset_to_params, x0, bounds = fit.setup_model_params(df_dataset_key, df_params=df_params, noise_models=noise_models)

x = np.zeros_like(x0)
for p in param_to_index:
    x[param_to_index[p]] = df_params.query("name==@p").iloc[0]['val']

args = (df_dataset_key, df_data, prefit_params, param_to_index, dataset_to_params, noise_models)

fit.predict(x, args, df_data)

display(df_data)

print(len(df_data))
print(len(df_data.dropna()))



df_data['phospho_GFP_denoise'] = 0.0
df_data['phospho_GFP_predict'] = 0.0
df_data['phospho_anti_predict'] = 0.0
df_data['kinase2_phospho_GFP_denoise'] = 0.0
df_data['kinase2_phospho_GFP_predict'] = 0.0
df_data['kinase2_phospho_anti_predict'] = 0.0


for exp_name, row in df_dataset_key.iterrows():
    
    phospho_factor = row['substrate_phospho_factor']
    kinase2_phospho_factor = row['kinase2_phospho_factor']

    df_tmp = df_data.query("exp_name==@exp_name")
    
    phospho_sigma = df_params.query("name=='phospho_sigma'")['val'].values[0]

    MOCU = phospho_factor*df_data.loc[df_tmp.index, 'phospho_MOCU_predict']
    
    df_data.loc[df_tmp.index, 'phospho_GFP_denoise'] = MOCU + noise_models['phospho']['anti_as_GFP_background'].mean
        
    df_data.loc[df_tmp.index, 'phospho_GFP_predict'] = noise_models['phospho']['GFP2MOCU'].sample(MOCU, phospho_sigma)

    df_data.loc[df_tmp.index, 'phospho_anti_predict'] = noise_models['phospho']['anti2GFP'].inverse_transform(df_data.loc[df_tmp.index, 'phospho_GFP_predict'])

    
    kinase2_phospho_sigma = df_params.query("name=='kinase2_phospho_sigma'")['val'].values[0]

    MOCU = kinase2_phospho_factor*df_data.loc[df_tmp.index, 'kinase2_phospho_MOCU_predict']
    
    df_data.loc[df_tmp.index, 'kinase2_phospho_GFP_denoise'] = MOCU + noise_models['kinase2_phospho']['anti_as_GFP_background'].mean
        
    df_data.loc[df_tmp.index, 'kinase2_phospho_GFP_predict'] = noise_models['kinase2_phospho']['GFP2MOCU'].sample(MOCU, kinase2_phospho_sigma)

    df_data.loc[df_tmp.index, 'kinase2_phospho_anti_predict'] = noise_models['kinase2_phospho']['anti2GFP'].inverse_transform(df_data.loc[df_tmp.index, 'kinase2_phospho_GFP_predict'])

    
display(df_data)


df_data.to_csv("../data/"+label+"/model_predictions.csv", sep=',')