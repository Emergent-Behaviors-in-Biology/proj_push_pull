import numpy as np
import numpy.random as rand
import pandas as pd
import scipy.optimize as opt


import seaborn as sns
import matplotlib.pyplot as plt


class MixtureNoise:
    
    def __init__(self, nonempty, empty):
        
        # empirical empty models for nonempty and empty
        self.nonempty = nonempty
        self.empty = empty
        
        
    def calc_mixture(self, anti, plot=False):
        
        prob_nonempty = self.nonempty.calc_prob_anti(anti)
        prob_empty = self.empty.calc_prob_anti(anti)
 
        
        def func(frac):
            f = -np.sum(np.log((1-frac)*prob_nonempty+frac*prob_empty+1e-8)) / len(anti)
#             print(frac, f)
            return f
            
        
        res = opt.minimize_scalar(func, bounds=(0, 1), method='bounded')
        
        print(res)
        
        frac = res.x
        
        
        return frac
        
    def anti_to_GFP(self, anti, fraction):
        
        # calculate mixture ratio
        
        # resample from mixture
        
        pass
    

    
class EmpiricalNoise:
    
    def __init__(self, fname, label_anti, label_GFP, nbins_anti=100, nbins_GFP=100, verbose=False):
        
        self.label_anti = label_anti
        self.label_GFP = label_GFP
        self.nbins_anti = nbins_anti
        self.nbins_GFP = nbins_GFP
        self.df = pd.read_csv(fname)
        
        self.df = self.df[(self.df[self.label_anti] > 0.0) & (self.df[self.label_GFP] > 0.0)]
        
        if verbose:
            display(self.df)
    
        x = np.log10(self.df[self.label_anti])
        y = np.log10(self.df[self.label_GFP])
        hist, edges_anti, edges_GFP = np.histogram2d(x, y, 
                                        bins=(self.nbins_anti, self.nbins_GFP))
        
        
        norm = hist.sum()
        self.prob_joint = hist / norm
        self.prob_anti = hist.sum(axis=1) / norm
        self.prob_GFP = hist.sum(axis=0) / norm
        
        self.edges_anti = edges_anti
        self.edges_GFP = edges_GFP
    
    
    def plot(self, ax, color='b', cbar=True):
        
        
        sns.histplot(self.df, x=self.label_GFP, y=self.label_anti, 
                              bins=(self.nbins_GFP, self.nbins_anti), 
                         log_scale=(True, True), cbar=cbar, ax=ax, color=color)
        
        
    def calc_prob_anti(self, anti):
        
        min_anti = 10**self.edges_anti[0]
        max_anti = 10**self.edges_anti[-1]
        
        bins_anti = np.digitize(anti, 10**self.edges_anti, right=False)-1
        
        bins_anti[anti >= max_anti] = -1
        bins_anti[anti < min_anti] = -1
        
        prob = np.zeros_like(anti)        
        prob[bins_anti != -1] = self.prob_anti[bins_anti[bins_anti != -1]]
        
        return prob
        
        

    def anti_to_GFP(self, anti, plot=False):
        
        min_anti = 10**self.edges_anti[0]
        max_anti = 10**self.edges_anti[-1]
        
        bins_anti = np.digitize(anti, 10**self.edges_anti, right=False)-1
        
        bins_anti[anti >= max_anti] = -1
        bins_anti[anti < min_anti] = -1
        
        unique_bins = np.unique(bins_anti)
        
        GFP_vals = 10**((self.edges_GFP[:self.nbins_GFP]
                         +self.edges_GFP[1:self.nbins_GFP+1])/2.0)
        
        GFP = np.full_like(anti, np.nan)
        for b in unique_bins:
                        
            if b == -1:
                continue
            
            idx = np.nonzero(bins_anti==b)[0] 
            
            if self.prob_anti[b] > 0.0:
                p = self.prob_joint[b]/self.prob_anti[b]
                GFP[idx] = rand.choice(GFP_vals, size=len(idx), p=p)
            
            
        if plot:
            
            ax = sns.histplot(x=GFP, y=anti, 
                              bins=(self.edges_GFP, self.edges_anti), 
                         log_scale=(True, True), cbar=True)
            
            ax.set_xlabel("GFP")
            ax.set_ylabel("antibody")

            plt.show()
            
        
        return GFP
    
    
    def GFP_to_anti(self, GFP, plot=False):
        
        min_GFP = 10**self.edges_GFP[0]
        max_GFP = 10**self.edges_GFP[-1]
        
        bins_GFP = np.digitize(GFP, 10**self.edges_GFP, right=False)-1
        
        bins_GFP[GFP >= max_GFP] = -1
        bins_GFP[GFP < min_GFP] = -1
        
        unique_bins = np.unique(bins_GFP)
        
        anti_vals = 10**((self.edges_anti[:self.nbins_anti]
                         +self.edges_anti[1:self.nbins_anti+1])/2.0)
        
        anti = np.full_like(GFP, np.nan)
        for b in unique_bins:
                        
            if b == -1:
                continue
            
            idx = np.nonzero(bins_GFP==b)[0] 
            
            if self.prob_GFP[b] > 0.0:
                p = self.prob_joint[:, b]/self.prob_GFP[b]
                anti[idx] = rand.choice(anti_vals, size=len(idx), p=p)
            
            
        if plot:
            
            ax = sns.histplot(x=GFP, y=anti, 
                              bins=(self.edges_GFP, self.edges_anti), 
                         log_scale=(True, True), cbar=True)
            
            ax.set_xlabel("GFP")
            ax.set_ylabel("antibody")

            plt.show()
            
        
        return anti
    
    
    
    def GFP_to_anti_max(self, GFP, plot=False):
        
        min_GFP = 10**self.edges_GFP[0]
        max_GFP = 10**self.edges_GFP[-1]
        
        bins_GFP = np.digitize(GFP, 10**self.edges_GFP, right=False)-1
        
        bins_GFP[GFP >= max_GFP] = -1
        bins_GFP[GFP < min_GFP] = -1
        
        unique_bins = np.unique(bins_GFP)
        
        anti_vals = 10**((self.edges_anti[:self.nbins_anti]
                         +self.edges_anti[1:self.nbins_anti+1])/2.0)
        
        anti = np.full_like(GFP, np.nan)
        for b in unique_bins:
                        
            if b == -1:
                continue
            
            idx = np.nonzero(bins_GFP==b)[0] 
            
            if self.prob_GFP[b] > 0.0:
                p = self.prob_joint[:, b]/self.prob_GFP[b]
                anti[idx] = anti_vals[np.argmax(p)]
                        
        return anti


# class LogNormNoiseModel:
    
#     def __init__(self, mean=None, cov=None, Sigma2=None, A=None, B=None):
        
#         if mean is not None and cov is not None:
#             # mean and covariance matrix between antibody and GFP measurements
#             # first element should be antibody units
#             # second element should be GFP
#             self.mean = mean
#             self.cov = cov

#             # correlation coeff
#             self.rho = cov[0, 1]/np.sqrt(cov[0,0])/np.sqrt(cov[1,1])

#             # antibody noise model parameters
#             self.Sigma2_anti = cov[0,0]*(1-self.rho)
#             self.A_anti = self.rho*cov[0,0]/cov[1,1]
#             self.B_anti = mean[0] - self.A_anti*mean[1]

#             print("Antibody Noise Model: Sigma^2 {0:.2f} A {1:.2f} B {2:.2f}".format(self.Sigma2_anti, self.A_anti, self.B_anti))

#             # GFP noise model parameters
#             self.Sigma2_GFP = cov[1,1]*(1-self.rho)
#             self.A_GFP = self.rho*cov[1,1]/cov[0,0]
#             self.B_GFP = mean[1] - self.A_GFP*mean[0]

#             print("GFP Noise Model: Sigma^2 {0:.2f} A {1:.2f} B {2:.2f}".format(self.Sigma2_GFP, self.A_GFP, self.B_GFP))
#         else:
#             self.Sigma2_anti = Sigma2
#             self.A_anti = A
#             self.B_anti = B
            
#             self.Sigma2_GFP = Sigma2
#             self.A_GFP = A
#             self.B_GFP = B
        
#     def anti_to_GFP(self, anti_vals):
        
#         return 10**rand.normal(self.A_GFP*np.log10(anti_vals)+self.B_GFP, np.sqrt(self.Sigma2_GFP))
    
    
#     def GFP_to_anti(self, GFP_vals):
        
#         return 10**rand.normal(self.A_anti*np.log10(GFP_vals)+self.B_anti, np.sqrt(self.Sigma2_anti))
    


        