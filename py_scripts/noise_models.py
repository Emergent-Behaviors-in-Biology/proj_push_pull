import numpy as np
import numpy.random as rand

class LogNormNoiseModel:
    
    def __init__(self, mean=None, cov=None, Sigma2=None, A=None, B=None):
        
        if mean is not None and cov is not None:
            # mean and covariance matrix between antibody and GFP measurements
            # first element should be antibody units
            # second element should be GFP
            self.mean = mean
            self.cov = cov

            # correlation coeff
            self.rho = cov[0, 1]/np.sqrt(cov[0,0])/np.sqrt(cov[1,1])

            # antibody noise model parameters
            self.Sigma2_anti = cov[0,0]*(1-self.rho)
            self.A_anti = self.rho*cov[0,0]/cov[1,1]
            self.B_anti = mean[0] - self.A_anti*mean[1]

            print("Antibody Noise Model: Sigma^2 {0:.2f} A {1:.2f} B {2:.2f}".format(self.Sigma2_anti, self.A_anti, self.B_anti))

            # GFP noise model parameters
            self.Sigma2_GFP = cov[1,1]*(1-self.rho)
            self.A_GFP = self.rho*cov[1,1]/cov[0,0]
            self.B_GFP = mean[1] - self.A_GFP*mean[0]

            print("GFP Noise Model: Sigma^2 {0:.2f} A {1:.2f} B {2:.2f}".format(self.Sigma2_GFP, self.A_GFP, self.B_GFP))
        else:
            self.Sigma2_anti = Sigma2
            self.A_anti = A
            self.B_anti = B
            
            self.Sigma2_GFP = Sigma2
            self.A_GFP = A
            self.B_GFP = B
        
    def anti_to_GFP(self, anti_vals):
        
        return 10**rand.normal(self.A_GFP*np.log10(anti_vals)+self.B_GFP, np.sqrt(self.Sigma2_GFP))
    
    
    def GFP_to_anti(self, GFP_vals):
        
        return 10**rand.normal(self.A_anti*np.log10(GFP_vals)+self.B_anti, np.sqrt(self.Sigma2_anti))
    

# class EmpNoiseModel:
    
#     def __init__(self, fname, anti_label, GFP_label, nbins_anti=100, nbins_gfp=100, verbose=False):
        
#         self.anti_label = anti_label
#         self.GFP_label = GFP_label
#         self.nbins_anti = nbins_anti
#         self.nbins_gfp = nbins_gfp
#         self.df_noise = pd.read_csv(fname)
        
#         self.df_noise = self.df_noise[(self.df_noise[self.df_noise.columns] >= 0).all(axis=1)]
        
#         if verbose:
#             display(self.df_noise)
            
#             ax = sns.histplot(self.df_noise, x='GFP - Area', y='Flag Antibody', bins=(self.nbins_gfp, self.nbins_anti), 
#                          log_scale=(True, True), cbar=True)

#             plt.show()
    

#         hist, edges_anti, edges_anti_GFP = np.histogram2d(np.log10(self.df_noise[self.anti_label]), np.log10(self.df_noise[self.GFP_label]), 
#                                                                    bins=(self.nbins_anti, self.nbins_gfp))
        
#         self.hist = hist
#         self.edges_anti = edges_anti
#         self.edges_anti_GFP = edges_anti_GFP
    
    

#     def anti_to_GFP(anti):
        
#         bins_anti = np.digitize(anti, 10**self.edges_anti, right=True)
        
#         unique_bins = np.unique(bins_anti)
        
#         for b in unique_bins:
#             print(np.nonzero(bins==b))
            
#             norm = np.sum(hist_writer[int(WT_anti_bin)])
        

# df_sample['WT_anti_bin'] = pd.cut(df_sample['WT_anti'], bins=10**xedges_writer, labels=False)




# df_resample = pd.concat([df_sample for i in range(10)]).reset_index(drop=True)

# display(df_resample)

# df_resample['WT_GFP'] = -1

# for WT_anti_bin, group in df_resample.groupby(['WT_anti_bin']):
# #     print(WT_anti_bin)
    
#     norm = np.sum(hist_writer[int(WT_anti_bin)])
#     if norm > 0.0:
#         p = hist_writer[int(WT_anti_bin)] / norm
#     else:
#         continue
        
#     samples = rand.choice(10**((yedges_writer[:nbins_gfp]+yedges_writer[1:nbins_gfp+1])/2.0), size=len(group.index), p=p)
#     # choice the bins numbers, then 
    
#     df_resample.loc[group.index, 'WT_GFP'] = samples
