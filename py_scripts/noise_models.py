import numpy as np
import numpy.random as rand
import pandas as pd
import scipy.optimize as opt


import seaborn as sns
import matplotlib.pyplot as plt




def calc_mixture(anti, empty, nonempty, plot=False):
        
        
        def func(x):
            
            (frac, scale) = x
            

#             print(scale, np.log(anti.max())/scale)
            
            xmin = np.min([np.log10(empty.edges_anti[0]), np.log10(nonempty.edges_anti[0]), np.log10(anti.min()) - scale])
            xmax = np.max([np.log10(empty.edges_anti[-1]), np.log10(nonempty.edges_anti[-1]), np.log10(anti.max()) - scale])
            
            
            loganti_empty = np.log10(empty.get_anti())
            loganti_nonempty = np.log10(nonempty.get_anti())
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
            
            
            
            
        def callback(x):
            print(func(x), x)

        res = opt.minimize(func, (0, 0), method='L-BFGS-B', 
                       jac='2-point', bounds=[(0, 1), (None, None)], callback=callback,
                          options={'finite_diff_rel_step': 1e-2})
                
        
        print(res)
        
        frac = res.x[0]
        scale = res.x[1]
        
        return (frac, scale)


# class MixtureNoise:
    
#     def __init__(self, nonempty, empty):
        
#         # empirical empty models for nonempty and empty
#         self.nonempty = nonempty
#         self.empty = empty
     
#     def calc_mixture(self, anti, plot=False):
        
        
#         def func(x):
            
#             (frac, offset) = x

#             xmin = np.min([self.empty.edges_anti[0], self.nonempty.edges_anti[0], np.log10(anti.min())-offset])
#             xmax = np.max([self.empty.edges_anti[-1], self.nonempty.edges_anti[-1], np.log10(anti.max())-offset])
                    
            
#             anti_empty = np.log10(self.empty.df[self.empty.label_anti])
#             anti_nonempty = np.log10(self.nonempty.df[self.nonempty.label_anti])
#             hist_noise, edges = np.histogram(np.concatenate([anti_empty, anti_nonempty]), 
#                                              bins=1000, range=(xmin, xmax), density=True,
#                                              weights=np.concatenate([frac*np.ones_like(anti_empty), 
#                                                    (1-frac)*np.ones_like(anti_nonempty)]))
            
#             cdf_infer = np.cumsum(hist_noise*(edges[1:]-edges[0:len(edges)-1]))
                        
#             hist_exp, edges = np.histogram(np.log10(anti)-offset, bins=1000, range=(xmin, xmax), density=True)
            
#             cdf_exp = np.cumsum(hist_exp*(edges[1:]-edges[0:len(edges)-1]))
                       
#             loss = np.max(np.abs(cdf_infer - cdf_exp))
            
# #             print(loss, x)
                
#             return loss
            
            
            
            
#         def callback(x):
#             print(func(x), x)

#         res = opt.minimize(func, (0, 0), method='L-BFGS-B', 
#                        jac='2-point', bounds=[(0, 1), (None, None)], callback=callback,
#                           options={'finite_diff_rel_step': 1e-2})
        
# #         res = opt.minimize_scalar(func, bounds=(0, 1), method='bounded')
        
        
#         print(res)
        
#         self.frac = res.x[0]
#         self.offset = res.x[1]
        
#         return (self.frac, self.offset)
    
    
#     def choose_empty(self, anti):
        
#         prob_nonempty = (1-self.frac)*self.nonempty.calc_prob_anti(anti)
#         prob_empty = self.frac*self.empty.calc_prob_anti(anti)
        
#         total = prob_nonempty + prob_empty
        
#         nonzero = total > 0.0
        
#         prob_nonempty[nonzero] /= total[nonzero]
#         prob_empty[nonzero] /= total[nonzero]
        
        
#         is_empty = np.full_like(anti, np.nan)
        
#         is_empty[nonzero] = np.where(rand.random(np.sum(nonzero)) < prob_empty[nonzero], 1, 0)
                
#         return is_empty
        
#     def anti_to_GFP(self, anti):
        
#         is_empty = self.choose_empty(anti)
        
#         GFP = np.full_like(anti, np.nan)
        
#         GFP[is_empty==1] = self.empty.anti_to_GFP(anti[is_empty==1])
#         GFP[is_empty==0] = self.nonempty.anti_to_GFP(anti[is_empty==0])
        
        
#         return is_empty, GFP
    

    
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
    
        hist, edges_loganti, edges_logGFP = np.histogram2d(np.log10(self.get_anti()), np.log10(self.get_GFP()), 
                                        bins=(self.nbins_anti, self.nbins_GFP))
        
        self.edges_anti = 10**edges_loganti
        self.edges_GFP = 10**edges_logGFP
        
        
        norm = hist.sum()
        self.prob_joint = hist / norm
        self.prob_anti = hist.sum(axis=1) / norm
        self.prob_GFP = hist.sum(axis=0) / norm
        
    def get_anti(self):
        return self.df[self.label_anti]
    
    def get_GFP(self):
        return self.df[self.label_GFP]
    
    
    def plot(self, ax, color='b', cbar=True):
        
        
        sns.histplot(self.df, x=self.label_GFP, y=self.label_anti, 
                              bins=(self.nbins_GFP, self.nbins_anti), 
                         log_scale=(True, True), cbar=cbar, ax=ax, color=color)
        

        anti_vals = (self.edges_anti[:self.nbins_anti]
                         +self.edges_anti[1:self.nbins_anti+1])/2.0
        
        ax.plot(self.anti_to_GFP_avg(anti_vals), anti_vals, 'k-')
        
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
        
        

#     def anti_to_GFP(self, anti, plot=False):
        
#         min_anti = 10**self.edges_anti[0]
#         max_anti = 10**self.edges_anti[-1]
        
#         bins_anti = np.digitize(anti, 10**self.edges_anti, right=False)-1
        
#         bins_anti[anti >= max_anti] = -1
#         bins_anti[anti < min_anti] = -1
        
#         unique_bins = np.unique(bins_anti)
        
#         GFP_vals = 10**((self.edges_GFP[:self.nbins_GFP]
#                          +self.edges_GFP[1:self.nbins_GFP+1])/2.0)
        
#         GFP = np.full_like(anti, np.nan)
#         for b in unique_bins:
                        
#             if b == -1:
#                 continue
            
#             idx = bins_anti==b
            
#             if self.prob_anti[b] > 0.0:
#                 p = self.prob_joint[b]/self.prob_anti[b]
#                 GFP[idx] = rand.choice(GFP_vals, size=np.sum(idx), p=p)
            
            
#         if plot:
            
#             ax = sns.histplot(x=GFP, y=anti, 
#                               bins=(self.edges_GFP, self.edges_anti), 
#                          log_scale=(True, True), cbar=True)
            
#             ax.set_xlabel("GFP")
#             ax.set_ylabel("antibody")

#             plt.show()
            
        
#         return GFP
    
    
#     def GFP_to_anti(self, GFP, plot=False):
        
#         min_GFP = 10**self.edges_GFP[0]
#         max_GFP = 10**self.edges_GFP[-1]
        
#         bins_GFP = np.digitize(GFP, 10**self.edges_GFP, right=False)-1
        
#         bins_GFP[GFP >= max_GFP] = -1
#         bins_GFP[GFP < min_GFP] = -1
        
#         unique_bins = np.unique(bins_GFP)
        
#         anti_vals = 10**((self.edges_anti[:self.nbins_anti]
#                          +self.edges_anti[1:self.nbins_anti+1])/2.0)
        
#         anti = np.full_like(GFP, np.nan)
#         for b in unique_bins:
                        
#             if b == -1:
#                 continue
            
#             idx = np.nonzero(bins_GFP==b)[0] 
            
#             if self.prob_GFP[b] > 0.0:
#                 p = self.prob_joint[:, b]/self.prob_GFP[b]
#                 anti[idx] = rand.choice(anti_vals, size=len(idx), p=p)
            
            
#         if plot:
            
#             ax = sns.histplot(x=GFP, y=anti, 
#                               bins=(self.edges_GFP, self.edges_anti), 
#                          log_scale=(True, True), cbar=True)
            
#             ax.set_xlabel("GFP")
#             ax.set_ylabel("antibody")

#             plt.show()
            
        
#         return anti
    
    
    
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