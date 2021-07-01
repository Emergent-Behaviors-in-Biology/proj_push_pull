import numpy as np
import numpy.random as rand
import pandas as pd
import scipy.optimize as opt


import seaborn as sns
import matplotlib.pyplot as plt




def calc_mixture(anti, empty_noise, nonempty_noise, plot=False):
        
        
        def func(x):
            
            (frac, scale) = x
            

#             print(scale, np.log(anti.max())/scale)
            
    
    
            xmin = np.min([np.log10(empty_noise.edges_anti[0]), np.log10(nonempty_noise.edges_anti[0]), np.log10(anti.min()) - scale])
            xmax = np.max([np.log10(empty_noise.edges_anti[-1]), np.log10(nonempty_noise.edges_anti[-1]), np.log10(anti.max()) - scale])
            
            
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
                
            return loss
            
            
            
            
#         def callback(x):
#             print(func(x), x)

#         res = opt.minimize(func, (0.05, 0.0), method='L-BFGS-B', 
#                        jac='2-point', bounds=[(0, 1), (None, None)], callback=callback,
#                           options={'finite_diff_rel_step': 1e-8, 'eps': 1e-4})
                
        def callback(x, f, accept):
            print(f, x, accept)
            
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


        res = opt.dual_annealing(func,bounds=[(0, 1), (-1, 1)], callback=callback, maxiter=2000,
                                no_local_search=True, x0=(0.0, 0.0))
        
        print(res)
        
        frac = res.x[0]
        scale = res.x[1]
        
        return (frac, scale)

    
def calc_prob_empty(anti, frac_empty, empty_noise, nonempty_noise):
    
    # if there are no empty cells
    if frac_empty == 0.0:
        return np.zeros_like(anti)
    
    # histograms of two data sets (density)
    # calc ratio
    # digitize to bins
    # for each data point, assign prob of empty
    
#     print(anti.min(), anti.max())
#     print(empty_noise.edges_anti.min(), empty_noise.edges_anti.max())
    
    # map each data point to a bin in the empty cell noise model and the nonempty cell noise model
    bins_empty = np.digitize(anti, empty_noise.edges_anti, right=False)-1
#     print(anti.values[bins_empty==-1])
#     print(anti[bins_empty==len(empty_noise.edges_anti)-1])
#     bins_empty[bins_empty==-1] = 0
#     bins_empty[bins_empty==len(empty_noise.edges_anti)-1] = len(empty_noise.edges_anti)-2
    
    bins_nonempty = np.digitize(anti, nonempty_noise.edges_anti, right=False)-1
#     print(anti.values[bins_nonempty==-1])
#     print(anti[bins_nonempty==len(nonempty_noise.edges_anti)-1])
#     bins_nonempty[bins_nonempty==-1] = 0
#     bins_nonempty[bins_nonempty==len(nonempty_noise.edges_anti)-1] = len(nonempty_noise.edges_anti)-2
    
    prob_empty = np.zeros_like(anti)
    
    # if data point is less than minimum empty cell noise bin, assume is noise
    prob_empty[bins_empty==-1] = 1.0
    
    # if empty cell data exists, calculate ratio
    idx = (bins_empty >=0) & (bins_empty < len(nonempty_noise.edges_anti)-1)
    
    pE = empty_noise.prob_anti[bins_empty[idx]]
    pNE = nonempty_noise.prob_anti[bins_nonempty[idx]]
    
    prob_empty[idx] = frac_empty*pE / (frac_empty*pE + (1-frac_empty)*pNE)
    
    return prob_empty

    
    
class EmpiricalNoise:
    
    def __init__(self, fname, label_anti, label_GFP, nbins_anti=100, nbins_GFP=100, verbose=False):
        
        self.nbins_anti = nbins_anti
        self.nbins_GFP = nbins_GFP
        self.df = pd.read_csv(fname)
        
        self.df = self.df[(self.df[label_anti] > 0.0) & (self.df[label_GFP] > 0.0)]
        self.df = self.df[[label_anti, label_GFP]].rename({label_anti: "anti", label_GFP: "GFP"}, axis=1)
        
        
        if verbose:
            display(self.df)
    
        self.calc_hist()
        
        
    def add_cells(self, noise2):
        
        self.df = pd.concat([self.df, noise2.df])
        
#         display(self.df)
        
        self.calc_hist()
        
        
    def calc_hist(self):
        
        hist, edges_loganti, edges_logGFP = np.histogram2d(np.log10(self.get_anti()), np.log10(self.get_GFP()), 
                                        bins=(self.nbins_anti, self.nbins_GFP))
        
        self.edges_anti = 10**edges_loganti
        self.edges_GFP = 10**edges_logGFP
        
        
        norm = hist.sum()
        self.prob_joint = hist / norm
        self.prob_anti = hist.sum(axis=1) / norm
        self.prob_GFP = hist.sum(axis=0) / norm
        
    def get_anti(self):
        return self.df['anti']
    
    def get_GFP(self):
        return self.df['GFP']
    
    
    def plot(self, ax, color='b', cbar=True):
        
        
        sns.histplot(self.df, x='GFP', y='anti', 
                              bins=(self.nbins_GFP, self.nbins_anti), 
                         log_scale=(True, True), cbar=cbar, ax=ax, color=color)
        

        anti_vals = (self.edges_anti[:self.nbins_anti]
                         +self.edges_anti[1:self.nbins_anti+1])/2.0
        
#         ax.plot(self.anti_to_GFP_avg(anti_vals), anti_vals, 'k-')
        ax.plot(self.anti_to_GFP_median(anti_vals), anti_vals, 'k-')
        
#         GFP_vals = (self.edges_GFP[:self.nbins_GFP]
#                          +self.edges_GFP[1:self.nbins_GFP+1])/2.0
        
#         ax.plot(GFP_vals, self.GFP_to_anti_avg(GFP_vals), 'r-')
        
        
#     def calc_prob_anti(self, anti):
        
#         min_anti = 10**self.edges_anti[0]
#         max_anti = 10**self.edges_anti[-1]
        
#         bins_anti = np.digitize(anti, 10**self.edges_anti, right=False)-1
        
#         bins_anti[anti >= max_anti] = -1
#         bins_anti[anti < min_anti] = -1
        
#         prob = np.full_like(anti, np.nan)        
#         prob[bins_anti != -1] = self.prob_anti[bins_anti[bins_anti != -1]]
        
#         return prob
        
        

    def anti_to_GFP(self, anti, plot=False):
        
        bins_anti = np.digitize(anti, self.edges_anti, right=False)-1
        
        unique_bins = np.unique(bins_anti)
        
        GFP_vals = np.sqrt(self.edges_GFP[:self.nbins_GFP]*self.edges_GFP[1:self.nbins_GFP+1])
        
        GFP = np.full_like(anti, np.nan)
        for b in unique_bins:
                        
            if b == -1 or b == len(self.edges_anti)-1:
#                 print(b)
                continue
            
            idx = bins_anti==b
            
            if self.prob_anti[b] > 0.0:
                p = self.prob_joint[b]/self.prob_anti[b]
                GFP[idx] = rand.choice(GFP_vals, size=np.sum(idx), p=p)
            
            
        if plot:
            
            ax = sns.histplot(x=GFP, y=anti, 
                              bins=(self.edges_GFP, self.edges_anti), 
                         log_scale=(True, True), cbar=True)
            
            ax.set_xlabel("GFP")
            ax.set_ylabel("antibody")

            plt.show()
            
        
        return GFP
    
    
    def GFP_to_anti(self, GFP, plot=False):
        
        bins_GFP = np.digitize(GFP, self.edges_GFP, right=False)-1
        
        unique_bins = np.unique(bins_GFP)
        
        anti_vals = np.sqrt(self.edges_anti[:self.nbins_anti]*self.edges_anti[1:self.nbins_anti+1])
        
        anti = np.full_like(GFP, np.nan)
        for b in unique_bins:
                        
            if b == -1 or b == len(self.edges_GFP)-1:
                continue
            
            idx = bins_GFP==b
            
            if self.prob_GFP[b] > 0.0:
                p = self.prob_joint[:, b]/self.prob_GFP[b]
                anti[idx] = rand.choice(anti_vals,  size=np.sum(idx), p=p)
            
            
        if plot:
            
            ax = sns.histplot(x=GFP, y=anti, 
                              bins=(self.edges_GFP, self.edges_anti), 
                         log_scale=(True, True), cbar=True)
            
            ax.set_xlabel("GFP")
            ax.set_ylabel("antibody")

            plt.show()
            
        
        return anti
    
    
    
    def anti_to_GFP_avg(self, anti):
                
        min_anti = self.edges_anti[0]
        max_anti = self.edges_anti[-1]
        
        bins_anti = np.digitize(anti, self.edges_anti, right=False)-1
        
        bins_anti[anti >= max_anti] = -1
        bins_anti[anti < min_anti] = -1
        
        unique_bins = np.unique(bins_anti)
        
        GFP_vals = (self.edges_GFP[:self.nbins_GFP]
                         +self.edges_GFP[1:self.nbins_GFP+1])/2.0
        
        GFP_avg = np.full_like(anti, np.nan)
        for b in unique_bins:
                        
            if b == -1:
                continue
            
            idx = bins_anti==b
            
            if self.prob_anti[b] > 0.0:
                GFP_avg[idx] = np.average(GFP_vals, weights=self.prob_joint[b])
            
            
        
        return GFP_avg
    
    
    def GFP_to_anti_avg(self, GFP, plot=False):
        
        min_GFP = self.edges_GFP[0]
        max_GFP = self.edges_GFP[-1]
        
        bins_GFP = np.digitize(GFP, self.edges_GFP, right=False)-1
        
        bins_GFP[GFP >= max_GFP] = -1
        bins_GFP[GFP < min_GFP] = -1
        
        unique_bins = np.unique(bins_GFP)
        
#         print(unique_bins)
        
        anti_vals = (self.edges_anti[:self.nbins_anti]
                         +self.edges_anti[1:self.nbins_anti+1])/2.0
        
        anti_avg = np.full_like(GFP, np.nan)
        for b in unique_bins:
                        
            if b == -1:
                continue
            
            idx = np.nonzero(bins_GFP==b)[0] 
                        
            if self.prob_GFP[b] > 0.0:
                anti_avg[idx] = np.average(anti_vals, weights=self.prob_joint[:, b])
            
       
        return anti_avg
    
    
    def anti_to_GFP_median(self, anti):
                
        min_anti = self.edges_anti[0]
        max_anti = self.edges_anti[-1]
        
        bins_anti = np.digitize(anti, self.edges_anti, right=False)-1
        
        bins_anti[anti >= max_anti] = -1
        bins_anti[anti < min_anti] = -1
        
        unique_bins = np.unique(bins_anti)
        
        GFP_vals = (self.edges_GFP[:self.nbins_GFP]
                         +self.edges_GFP[1:self.nbins_GFP+1])/2.0
        
        GFP_avg = np.full_like(anti, np.nan)
        for b in unique_bins:
                        
            if b == -1:
                continue
            
            idx = bins_anti==b
            
            if self.prob_anti[b] > 0.0:
                
                med_idx = np.nonzero(np.cumsum(self.prob_joint[b]/np.sum(self.prob_joint[b])) >= 0.5)[0][0]
                
                GFP_avg[idx] = GFP_vals[med_idx]
            
            
        
        return GFP_avg