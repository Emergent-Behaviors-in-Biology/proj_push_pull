from IPython.display import display, Markdown

import numpy as np
import numpy.random as rand
import pandas as pd
import scipy.optimize as opt


import seaborn as sns
import matplotlib.pyplot as plt


class RandomConditionalNoise:
    def __init__(self, in_data, out_data, ppbin=10, verbose=False):
                
        self.df = pd.DataFrame(np.c_[in_data, out_data], columns=['in', 'out'])
        
        self.ppbin = ppbin
        
        
        self.df = self.df[(self.df['in'] > 0.0) & (self.df['out'] > 0.0)]

        self.verbose = verbose
        
        if verbose:
            display(self.df)
    
        self.calc_hist()
        
    def get_in_data(self):
        return self.df['in']
    
    def get_out_data(self):
        return self.df['out']
    
    
    def plot(self, ax, color='b', cbar=True, percent_curves=False):
        
        
        sns.histplot(self.df, x='in', y='out', 
                              bins=(self.nbins, self.nbins), 
                         log_scale=(True, True), cbar=cbar, ax=ax, color=color)
        
        if percent_curves:
            for b in range(0, self.nbins, self.nbins//8):
                ax.plot(self.in_median, self.out_median[:, b], 'k-', lw=1.0)

        
        
    def add_cells(self, noise_model):
        
        self.df = pd.concat([self.df, noise_model.df])
        self.df.reset_index(drop=True, inplace=True)
#         display(self.df)
        
        self.calc_hist()
        
        
    def calc_hist(self):
        
        self.ncells = len(self.df.index)
        self.nbins = int(np.sqrt(self.ncells / self.ppbin))
        
        if self.verbose:
            print("Num Cells:", self.ncells, "Points per bin:", self.ppbin, "Num Bins:", self.nbins)
        
        bin_index, bin_edges = pd.qcut(self.get_in_data(), self.nbins,  labels=False, retbins=True)
        self.df['in_bin_index'] = bin_index
        self.in_bin_edges = bin_edges
        self.in_median = self.df.groupby('in_bin_index')['in'].median().values
        
        self.prob_anti = 1.0 / (bin_edges[1:] - bin_edges[:-1]) / self.nbins
                
        self.df['out_bin_index'] = -1
        self.out_bin_edges = np.zeros([self.nbins, self.nbins+1])
        self.out_median = np.zeros([self.nbins, self.nbins])
        for in_bin_index, group in self.df.groupby('in_bin_index'):
            bin_index, bin_edges = pd.qcut(group['out'], self.nbins,  labels=False, retbins=True)
                    
            self.df.loc[group.index, 'out_bin_index'] = bin_index
            self.out_bin_edges[in_bin_index] = bin_edges
            self.out_median[in_bin_index] = self.df.loc[group.index].groupby('out_bin_index')['out'].median().values
            
    
#         print(self.GFP_bin_edges)
#         print(self.GFP_median)
    
    def get_bin_index(self, in_vals):
                
        bin_index = np.digitize(in_vals, self.in_bin_edges, right=False)-1

#         bin_index[bin_index==-1] = -1
        # this same as len(self.anti_bin_edges) - 1
        bin_index[bin_index==self.nbins] = -1
        
                
        return bin_index
    
        
    def transform(self, in_vals):
        
        in_bin_index = self.get_bin_index(in_vals)
        
        unique_bins = np.unique(in_bin_index)
                
        out_bin_index= np.full_like(in_bin_index, -1)
        out_vals = np.full_like(in_bin_index, -1.0, dtype=float)
        
        for b in unique_bins:
            
            idx = in_bin_index==b
            
            if b == -1:
                out_vals[idx] = np.nan
                continue
            
            
            out_bin_index[idx] = rand.randint(0, self.nbins, size=np.sum(idx))
            out_vals[idx] = self.out_median[b, out_bin_index[idx]]
                        
        return out_vals
    
    def resample_in(self, n_vals):
        return rand.choice(self.df['in'].values, size=n_vals)
    
    def resample_out(self, n_vals):
        return rand.choice(self.df['out'].values, size=n_vals)
        

    
    
class PercentileNoise:
    def __init__(self, in_data, out_data, nbins, verbose=False):
        
        self.in_data = in_data[in_data > 0.0]
        self.out_data = out_data[out_data > 0.0]
        
        self.nbins = nbins
        
        
        in_percentiles = np.percentile(in_data, np.linspace(0, 100, nbins+1))
        out_percentiles = np.percentile(out_data, np.linspace(0, 100, nbins+1))
        
        self.df = pd.DataFrame(np.c_[np.linspace(0, 100, nbins+1), in_percentiles, out_percentiles], columns=['percentile', 'in', 'out'])
        
        self.out_mean = np.sqrt(out_percentiles[0:nbins]*out_percentiles[1:nbins+1])
        self.in_mean = np.sqrt(in_percentiles[0:nbins]*in_percentiles[1:nbins+1])
        
        
    def plot(self, ax, color='b'):
        
        ax.scatter(self.df['in'], self.df['out'], s=1+self.df['percentile'])
        ax.plot(self.df['in'], self.df['out'], color='k')
        ax.scatter(self.in_mean, self.out_mean, s=20, color='r', marker='x')
        
    
    def get_in_data(self):
        return self.in_data
    
    def get_out_data(self):
        return self.out_data
    
    def get_bin_index(self, in_vals):
                            
        bin_index = np.digitize(in_vals, self.df['in'], right=False)-1

        bin_index[bin_index==self.nbins] = -1
        
                
        return bin_index
        
        
    
        
    def transform(self, in_vals):
        
        in_bin_index = self.get_bin_index(in_vals)
                
        out_vals = np.full_like(in_bin_index, -1.0, dtype=float)
        
        out_vals[in_bin_index==-1] = np.nan
        
        idx = in_bin_index!=-1
        out_vals[idx] = self.out_mean[in_bin_index[idx]]
                        
        return out_vals
    
    
    
    
    
    
    
    
    

class Anti2GFPNoise:
    def __init__(self, fname, anti_label, GFP_label, ppbin=10, verbose=False):
        
        self.ppbin = ppbin
        self.df = pd.read_csv(fname)
        
        
        self.df = self.df[(self.df[anti_label] > 0.0) & (self.df[GFP_label] > 0.0)]
        self.df = self.df[[anti_label, GFP_label]].rename({anti_label: "anti", GFP_label: "GFP"}, axis=1)

        
        if verbose:
            display(self.df)
    
        self.calc_hist()
        
    def get_anti(self):
        return self.df['anti']
    
    def get_GFP(self):
        return self.df['GFP']
    
    
    def plot(self, ax, color='b', cbar=True):
        
        
        sns.histplot(self.df, x='GFP', y='anti', 
                              bins=(self.nbins, self.nbins), 
                         log_scale=(True, True), cbar=cbar, ax=ax, color=color)
        
#         for b in range(0, self.nbins, self.nbins//8):
#             ax.plot(self.GFP_median[:, b], self.anti_median, 'k-', lw=1.0)

        
        
    def add_cells(self, noise2):
        
        self.df = pd.concat([self.df, noise2.df])
        self.df.reset_index(drop=True, inplace=True)
#         display(self.df)
        
        self.calc_hist()
        
        
    def calc_hist(self):
        
        self.ncells = len(self.df.index)
        self.nbins = int(np.sqrt(self.ncells / self.ppbin))
        
        print("Num Cells:", self.ncells, "Points per bin:", self.ppbin, "Num Bins:", self.nbins)
        
        out, bin_edges = pd.qcut(self.get_anti(), self.nbins,  labels=False, retbins=True)
        self.df['anti_bin'] = out
        self.anti_bin_edges = bin_edges
        self.anti_median = self.df.groupby('anti_bin')['anti'].median().values
        
        self.prob_anti = 1.0 / (bin_edges[1:] - bin_edges[:-1]) / self.nbins
                
        self.df['GFP_bin'] = -1
        self.GFP_bin_edges = np.zeros([self.nbins, self.nbins+1])
        self.GFP_median = np.zeros([self.nbins, self.nbins])
        for anti_bin, group in self.df.groupby('anti_bin'):
            out, bin_edges = pd.qcut(group['GFP'], self.nbins,  labels=False, retbins=True)
                    
            self.df.loc[group.index, 'GFP_bin'] = out
            self.GFP_bin_edges[anti_bin] = bin_edges
            self.GFP_median[anti_bin] = self.df.loc[group.index].groupby('GFP_bin')['GFP'].median().values
            
    
#         print(self.GFP_bin_edges)
#         print(self.GFP_median)
    
    def anti_to_bin(self, anti):
                
        anti_bins = np.digitize(anti, self.anti_bin_edges, right=False)-1

        anti_bins[anti_bins==-1] = -1
        # this same as len(self.anti_bin_edges) - 1
        anti_bins[anti_bins==self.nbins] = -1
        
                
        return anti_bins
    
        
    def anti_to_GFP(self, anti, plot=False):
        
        anti_bins = self.anti_to_bin(anti)
        
        unique_bins = np.unique(anti_bins)
                
        GFP_bins= np.full_like(anti_bins, -1)
        GFP = np.full_like(anti_bins, -1.0, dtype=float)
        
        for b in unique_bins:
            
            idx = anti_bins==b
            
            if b == -1:
                GFP[idx] = np.nan
                continue
            
            
            GFP_bins[idx] = rand.randint(0, self.nbins, size=np.sum(idx))
            GFP[idx] = self.GFP_median[b, GFP_bins[idx]]
                        
        return (GFP, anti_bins, GFP_bins)
    
    
class GFP2AntiNoise:
    def __init__(self, fname, GFP_label, anti_label, ppbin=10, verbose=False):
        
        self.ppbin = ppbin
        self.df = pd.read_csv(fname)
        
        
        self.df = self.df[(self.df[GFP_label] > 0.0) & (self.df[anti_label] > 0.0)]
        self.df = self.df[[GFP_label, anti_label]].rename({GFP_label: "GFP", anti_label: "anti"}, axis=1)

        
        if verbose:
            display(self.df)
    
        self.calc_hist()
        
    def get_GFP(self):
        return self.df['GFP']
    
    def get_anti(self):
        return self.df['anti']
    
    
    def plot(self, ax, color='b', cbar=True):
        
        
        sns.histplot(self.df, x='GFP', y='anti', 
                              bins=(self.nbins, self.nbins), 
                         log_scale=(True, True), cbar=cbar, ax=ax, color=color)
        
#         for b in range(0, self.nbins, self.nbins//8):
#             ax.plot(self.GFP_median, self.anti_median[:, b], 'k-', lw=1.0)

        
        
    def add_cells(self, noise2):
        
        self.df = pd.concat([self.df, noise2.df])
        self.df.reset_index(drop=True, inplace=True)
#         display(self.df)
        
        self.calc_hist()
        
        
    def calc_hist(self):
        
        self.ncells = len(self.df.index)
        self.nbins = int(np.sqrt(self.ncells / self.ppbin))
        
        print("Num Cells:", self.ncells, "Points per bin:", self.ppbin, "Num Bins:", self.nbins)
        
        out, bin_edges = pd.qcut(self.get_GFP(), self.nbins,  labels=False, retbins=True)
        self.df['GFP_bin'] = out
        self.GFP_bin_edges = bin_edges
        self.GFP_median = self.df.groupby('GFP_bin')['GFP'].median().values
        
        self.prob_GFP = 1.0 / (bin_edges[1:] - bin_edges[:-1]) / self.nbins
                
        self.df['anti_bin'] = -1
        self.anti_bin_edges = np.zeros([self.nbins, self.nbins+1])
        self.anti_median = np.zeros([self.nbins, self.nbins])
        for GFP_bin, group in self.df.groupby('GFP_bin'):
            out, bin_edges = pd.qcut(group['anti'], self.nbins,  labels=False, retbins=True)
                    
            self.df.loc[group.index, 'anti_bin'] = out
            self.anti_bin_edges[GFP_bin] = bin_edges
            self.anti_median[GFP_bin] = self.df.loc[group.index].groupby('anti_bin')['anti'].median().values
            
    
#         print(self.anti_bin_edges)
#         print(self.anti_median)
    
    def GFP_to_bin(self, GFP):
        
        GFP_bins = np.digitize(GFP, self.GFP_bin_edges, right=False)-1
        GFP_bins[GFP_bins==-1] = 0
        GFP_bins[GFP_bins==self.nbins] = self.nbins-1
        
        return GFP_bins
    
        
    def GFP_to_anti(self, GFP, plot=False):
        
        GFP_bins = self.GFP_to_bin(GFP)
        
        unique_bins = np.unique(GFP_bins)
        
        anti_bins= np.full_like(GFP_bins, -1)
        anti= np.full_like(GFP_bins, -1.0, dtype=float)
        for b in unique_bins:
            idx = GFP_bins==b
            
            anti_bins[idx] = rand.randint(0, self.nbins, size=np.sum(idx))
            anti[idx] = self.anti_median[b, anti_bins[idx]]
                 
        return (anti, GFP_bins, anti_bins)
        
     

counter = 0

def calc_mixture(anti, empty_noise, nonempty_noise, seed=42, maxiter=2000, plot=False):
        
        
        global counter
        counter = 0
        
        def func(x):
            
            (frac, scale) = x
            

#             print(scale, np.log(anti.max())/scale)
            
    
    
            xmin = np.min([np.log10(empty_noise.anti_bin_edges[0]), np.log10(nonempty_noise.anti_bin_edges[0]), np.log10(anti.min()) - scale])
            xmax = np.max([np.log10(empty_noise.anti_bin_edges[-1]), np.log10(nonempty_noise.anti_bin_edges[-1]), np.log10(anti.max()) - scale])
            
            loganti_empty = np.log10(empty_noise.get_anti())
            loganti_nonempty = np.log10(nonempty_noise.get_anti())
            hist_noise, edges = np.histogram(np.concatenate([loganti_empty, loganti_nonempty]), 
                                             bins=1000, range=(xmin, xmax), density=True,
                                             weights=np.concatenate([frac*np.ones_like(loganti_empty), 
                                                   (1-frac)*np.ones_like(loganti_nonempty)]))
            
            cdf_infer = np.cumsum(hist_noise*(edges[1:]-edges[0:len(edges)-1]))
                        
            hist_exp, edges = np.histogram(np.log10(anti) - scale, bins=1000, range=(xmin, xmax), density=True)
            
            cdf_exp = np.cumsum(hist_exp*(edges[1:]-edges[0:len(edges)-1]))
                       
            loss = np.max(np.abs(cdf_infer - cdf_exp))
            
#             print(loss, x)

            
            global counter
            counter += 1
                
            return loss
            
            
            
            
#         def callback(x):
#             print(func(x), x)

#         res = opt.minimize(func, (0.05, 0.0), method='L-BFGS-B', 
#                        jac='2-point', bounds=[(0, 1), (None, None)], callback=callback,
#                           options={'finite_diff_rel_step': 1e-8, 'eps': 1e-4})
                
        def callback(x, f, accept):
            print(counter, f, x, accept)
            
#         def accept_test(f_new, x_new, f_old, x_old):
#             if x_new[0] < 0.0 or x_new[0] > 1.0:
#                 return False
#             else:
#                 return True
            
#         res = opt.basinhopping(func, (0.0, 0.0), callback=callback, accept_test=accept_test,
#                                stepsize=0.01,
#                                minimizer_kwargs={'method': 'L-BFGS-B', 
#                                                  'options': {'eps': 1e-4},
#                                                  'bounds': [(0, 1), (None, None)]},
#                                niter=50)


        res = opt.dual_annealing(func,bounds=[(0, 0.25), (-1, 1)], callback=callback, maxiter=maxiter,
                                no_local_search=True, x0=(0.01, 0.0), seed=seed)
        
        print(res)
        
        frac = res.x[0]
        scale = res.x[1]
        
        return (frac, scale)

    
def calc_prob_empty(anti, frac_empty, empty_noise, nonempty_noise):
    
    # if there are no empty cells
    if frac_empty == 0.0:
        return np.zeros_like(anti)
    
    # map each data point to a bin in the empty cell noise model and the nonempty cell noise model
    bins_empty = np.digitize(anti, empty_noise.anti_bin_edges, right=False)-1

    bins_nonempty = np.digitize(anti, nonempty_noise.anti_bin_edges, right=False)-1

    prob_empty = np.zeros_like(anti)
    
    # if data point is less than minimum empty cell noise bin, assume is noise
    prob_empty[bins_empty==-1] = 1.0
    
    # if empty cell data exists, calculate ratio
    idx = (bins_empty >=0) & (bins_empty < nonempty_noise.nbins)
    
    pE = empty_noise.prob_anti[bins_empty[idx]]
    pNE = nonempty_noise.prob_anti[bins_nonempty[idx]]
    
    prob_empty[idx] = frac_empty*pE / (frac_empty*pE + (1-frac_empty)*pNE)
    
    return prob_empty

    
    
