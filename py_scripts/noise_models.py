from IPython.display import display, Markdown

import numpy as np
import scipy as sp
import numpy.random as rand
import numpy.linalg as la
import pandas as pd
import scipy.optimize as opt
import numpy.ma as ma
import matplotlib as mpl

import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.colors as mcolors

import seaborn as sns
import matplotlib.pyplot as plt

class Density:
    def __init__(self, data, ppbin=10):
        
        self.df = pd.DataFrame(np.c_[data], columns=['vals'])
        
        self.ppbin = ppbin
        
         
        self.ndata = len(self.df.index)
        self.nbins = self.ndata//self.ppbin
        
        self.median = np.median(self.df['vals'])
        self.mean = 10**np.mean(np.log10(self.df['vals']))
  
        bin_index, bin_edges = pd.qcut(self.df['vals'], self.nbins,  labels=False, retbins=True)
        self.df['bin_index'] = bin_index
        self.bin_edges = bin_edges
        
        self.density = 1.0 / (bin_edges[1:] - bin_edges[:-1]) / self.nbins
        self.density_logscale = 1.0 / (np.log10(bin_edges[1:]) - np.log10(bin_edges[:-1])) / self.nbins
        
    def get_data(self):
        return self.df['vals'].values
        
    def get_bin_index(self, vals):
                
        bin_index = np.digitize(vals, self.bin_edges, right=False)-1

        # this same as len(self.anti_bin_edges) - 1
        bin_index[bin_index==self.nbins] = -1
        
        return bin_index
        
    def get_density(self, vals, log_scale=False):
        
        bin_index = self.get_bin_index(vals)
        
        unique_bins = np.unique(bin_index)
                
        p = np.zeros_like(vals)
        
        for b in unique_bins:
            
            idx = bin_index==b
            
            if b == -1:
                continue
              
            if log_scale:
                p[idx] = self.density_logscale[b]
            else:
                p[idx] = self.density[b]
                        
        return p
        
        

class RandomConditionalNoise:
    def __init__(self, in_data, out_data, ppbin=10, verbose=False):
                
        self.df = pd.DataFrame(np.c_[in_data, out_data], columns=['in_data', 'out_data'])
        
        self.ppbin = ppbin
        
        
        self.verbose = verbose
        
        if verbose:
            display(self.df)
    
        self.calc_hist()
        
    def get_in_data(self):
        return self.df['in_data']
    
    def get_out_data(self):
        return self.df['out_data']
    
    
    def plot(self, ax, color='b', cbar=True):
        
        
        sns.histplot(self.df, x='in_data', y='out_data', 
                              bins=(self.nbins, self.nbins), 
                         log_scale=(True, True), cbar=cbar, ax=ax, color=color)

        
        
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
        self.df['in_data_bin_index'] = bin_index
        self.in_bin_edges = bin_edges
        self.in_median = self.df.groupby('in_data_bin_index')['in_data'].median().values
        
        self.prob_in_vals = 1.0 / (bin_edges[1:] - bin_edges[:-1]) / self.nbins
                
        self.df['out_data_bin_index'] = -1
        self.out_bin_edges = np.zeros([self.nbins, self.nbins+1])
        self.out_median = np.zeros([self.nbins, self.nbins])
        for in_bin_index, group in self.df.groupby('in_data_bin_index'):
            bin_index, bin_edges = pd.qcut(group['out_data'], self.nbins,  labels=False, retbins=True)
                    
            self.df.loc[group.index, 'out_data_bin_index'] = bin_index
            self.out_bin_edges[in_bin_index] = bin_edges
            self.out_median[in_bin_index] = self.df.loc[group.index].groupby('out_data_bin_index')['out_data'].median().values
            
    
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
    
    def get_prob(self, in_vals):
        
        in_bin_index = self.get_bin_index(in_vals)
        
        unique_bins = np.unique(in_bin_index)
                
        prob = np.zeros_like(in_vals)
        
        for b in unique_bins:
            
            idx = in_bin_index==b
            
            if b == -1:
                continue
                        
            prob[idx] = self.prob_in_vals[b]
                        
        return prob
        
    
    def resample_in(self, n_vals):
        return rand.choice(self.df['in_data'].values, size=n_vals)
    
    def resample_out(self, n_vals):
        return rand.choice(self.df['out_data'].values, size=n_vals)
      

def prob_normal(x, y, mu, Sigma):
        
    dx = x - mu[0]
    dy = y - mu[1]

    Sigmainv = la.inv(Sigma)
    detSigma = la.det(Sigma)

    return np.exp(-(Sigmainv[0, 0]*dx**2+Sigmainv[1, 1]*dy**2+2*Sigmainv[0, 1]*dx*dy)/2.0) / np.sqrt((2*np.pi)**2*detSigma)
        

    
def add_ellipse(ax, mean, cov):
    
    evals, evecs = la.eigh(cov)

    x = 10**(mean[0]+ np.linspace(-evecs[0, 0]*np.sqrt(evals[0]), evecs[0, 0]*np.sqrt(evals[0])))
    y = 10**(mean[1]+ np.linspace(-evecs[1, 0]*np.sqrt(evals[0]), evecs[1, 0]*np.sqrt(evals[0])))

    ax.plot(x, y, 'k-', lw=1.0)

    x = 10**(mean[0]+ np.linspace(-evecs[0, 1]*np.sqrt(evals[1]), evecs[0, 1]*np.sqrt(evals[1])))
    y = 10**(mean[1]+ np.linspace(-evecs[1, 1]*np.sqrt(evals[1]), evecs[1, 1]*np.sqrt(evals[1])))

    ax.plot(x, y, 'k-', lw=1.0)

    offset = mtransforms.ScaledTranslation(10**mean[0], 10**mean[1], ax.transScale)
    tform = offset + ax.transLimits + ax.transAxes
    ellipse = mpatches.Ellipse((0, 0), 2*np.sqrt(evals[0]), 2*np.sqrt(evals[1]), 
                               angle=np.arctan2(evecs[1, 0], evecs[0, 0])*180/np.pi, transform=tform,
                              fill=False, ec='k', lw=1.0)
    ax.add_patch(ellipse)

        
class GaussianConditionalNoise:
    def __init__(self, in_data, out_data, verbose=False):
                
        self.df = pd.DataFrame(np.c_[in_data, out_data], columns=['in_data', 'out_data'])
        
        self.mean = np.mean(np.log10(np.c_[in_data, out_data]), axis=0)
        self.cov = np.cov(np.log10(np.c_[in_data, out_data]), rowvar=False)
        
        
        
    def get_in_data(self):
        return self.df['in_data']
    
    def get_out_data(self):
        return self.df['out_data']
    
    
    def plot(self, ax, color='b', cbar=True):
        
                
        x = np.linspace(-2, 7, 200)
        y = np.linspace(-2, 7, 200)

        X, Y = np.meshgrid(x, y)
        Z = self.joint_prob(10**X, 10**Y)
        ax.pcolormesh(10**X, 10**Y, Z, cmap=plt.cm.Blues, rasterized=True, shading='auto')

        add_ellipse(ax, self.empty_mean, self.empty_cov)
        add_ellipse(ax, self.nonempty_mean, self.nonempty_cov)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        
    def get_joint_prob(self, in_vals, out_vals):
        
        return prob_normal(np.log10(in_vals), np.log10(out_vals), self.mean, self.cov)
    
    def get_conditional_prob(self, in_vals, out_vals):
        
        dx = np.log10(in_vals) - self.mean[0]
        
        loc = self.mean[1] + self.cov[0, 1] / self.cov[0, 0] * dx
        scale = self.cov[1, 1] - self.cov[0, 1]**2/self.cov[0, 0]
           
        dy = np.log10(out_vals) - loc
            
        return np.exp(-dy**2 / (2*scale)) / np.sqrt(2*np.pi*scale)
  
    def get_prob(self, in_vals):
        
        dx = np.log10(in_vals) - self.mean[0]
        
        return np.exp(-dx**2/(2*self.cov[0, 0])) / np.sqrt(2*np.pi*self.cov[0, 0])

    def transform(self, in_vals):
        
        dx = np.log10(in_vals) - self.mean[0]
        
        loc = self.mean[1] + self.cov[0, 1] / self.cov[0, 0] * dx
        scale = self.cov[1, 1] - self.cov[0, 1]**2/self.cov[0, 0]
        
        out_vals = 10**rand.normal(loc=loc, scale=scale)
                        
        return out_vals
    
class CompositeConditionalNoiseNoEmpty:
    def __init__(self, exp_noise, cutoff_percent=0.95):
                
        self.exp_noise = exp_noise
        self.gaussian_noise = GaussianConditionalNoise(exp_noise.get_in_data(), exp_noise.get_out_data())        
        
        self.cutoff_percent = cutoff_percent
    
        self.low_cutoff = np.quantile(self.exp_noise.get_in_data(), 1.0-self.cutoff_percent)
        self.high_cutoff = np.quantile(self.exp_noise.get_in_data(), self.cutoff_percent)
    
    
    def plot(self, ax, color='b'):
        
        self.exp_noise.plot(ax, color='g', cbar=False)
                
        
        ax.vlines([self.low_cutoff, self.high_cutoff], ymin=10**-2, ymax=10**7, color='r', ls='dashed')

        add_ellipse(ax, self.gaussian_noise.mean, self.gaussian_noise.cov)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlim(10**-2, 10**6)
        ax.set_ylim(10**-1, 10**7)
        
    def plot_composite(self, ax, color='b'):
        
        hist, xedges, yedges  = np.histogram2d(np.log10(self.exp_noise.get_in_data()), 
                                 np.log10(self.exp_noise.get_out_data()),
                                bins=100, range=[(np.log10(self.low_cutoff), np.log10(self.high_cutoff)), (-1, 7)], density=True)
        
       
        X, Y = np.meshgrid(xedges, yedges)
        
        norm = mcolors.Normalize(np.min(hist)/2, np.max(hist))
        cmap = plt.cm.Blues_r
        
        hist = ma.masked_equal(hist, 0)
        
        ax.pcolormesh(10**X, 10**Y, hist.T, cmap=cmap, norm=norm, rasterized=True, shading='auto')
        
        ax.vlines([self.low_cutoff, self.high_cutoff], ymin=10**-2, ymax=10**7, color='r', ls='dashed')

        add_ellipse(ax, self.gaussian_noise.mean, self.gaussian_noise.cov)        
        
        x = np.linspace(np.log10(self.high_cutoff), 6, 50)
        y = np.linspace(-1, 7, 200)

        X, Y = np.meshgrid(x, y)
        Z = self.gaussian_noise.get_joint_prob(10**X, 10**Y)
        
        Z = ma.masked_less(Z, np.min(hist)/2)
        ax.pcolormesh(10**X, 10**Y, Z, cmap=cmap, norm=norm, rasterized=True, shading='auto')
        
        
        x = np.linspace(-2, np.log10(self.low_cutoff), 50)
        y = np.linspace(-1, 7, 200)

        X, Y = np.meshgrid(x, y)
        Z = self.gaussian_noise.get_joint_prob(10**X, 10**Y)
        
        Z = ma.masked_less(Z, np.min(hist)/2)
        
        ax.pcolormesh(10**X, 10**Y, Z, cmap=cmap, norm=norm, rasterized=True, shading='auto')

        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlim(10**-2, 10**6)
        ax.set_ylim(10**-1, 10**7)
        
    def plot_conditional_prob(self, ax, cbar=False):
        
        
        hist, xedges, yedges  = np.histogram2d(np.log10(self.exp_noise.get_in_data()), 
                                 np.log10(self.exp_noise.get_out_data()),
                                bins=100, range=[(np.log10(self.low_cutoff), np.log10(self.high_cutoff)), (-1, 7)], density=False)
     
        prob_anti = np.sum(hist, axis=1)
   
        hist = hist / np.maximum(prob_anti[:, np.newaxis], 1.0)
        
        hist = ma.masked_equal(hist, 0)
        
        
        X, Y = np.meshgrid(xedges, yedges)
        
        norm = mcolors.LogNorm(np.min(hist), np.max(hist))
        cmap = plt.cm.Blues_r
        
        hist = ma.masked_equal(hist, 0)
        
        ax.pcolormesh(10**X, 10**Y, hist.T, cmap=cmap, norm=norm, rasterized=True, shading='auto')
        
        ax.vlines([self.low_cutoff, self.high_cutoff], ymin=10**-2, ymax=10**7, color='r', ls='dashed')
        
        
        x = np.linspace(np.log10(self.high_cutoff), 6, 50)
        y = np.linspace(-1, 7, 200)

        X, Y = np.meshgrid(x, y)
        
        
        prob = self.gaussian_noise.get_conditional_prob(10**X, 10**Y)
            
        Z = prob[:-1, :-1]
        dy = y[1:]-y[:-1]
        Z = Z * dy[:, np.newaxis]
        Z = ma.masked_less(Z, np.min(hist))
        
        ax.pcolormesh(10**X, 10**Y, Z, cmap=cmap, norm=norm, rasterized=True, shading='auto')
        
        
        x = np.linspace(-2, np.log10(self.low_cutoff), 50)
        y = np.linspace(-1, 7, 200)

        X, Y = np.meshgrid(x, y)
                
        prob = self.gaussian_noise.get_conditional_prob(10**X, 10**Y)
            
        Z = prob[:-1, :-1]
        dy = y[1:]-y[:-1]
        Z = Z * dy[:, np.newaxis]
        Z = ma.masked_less(Z, np.min(hist))
        
        ax.pcolormesh(10**X, 10**Y, Z, cmap=cmap, norm=norm, rasterized=True, shading='auto')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlim(10**-2, 10**6)
        ax.set_ylim(10**-1, 10**7)
        
        
        if cbar:
            
            bbox = ax.get_position()

            cax = ax.figure.add_axes([bbox.x1+0.02, bbox.y0, 0.02, bbox.y1-bbox.y0])
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')

            cax.tick_params(which='major', direction='out', length=3.0, width=1.0)

            cbar.set_label(r"Conditional Frequency")
          
    def get_prob(self, in_vals):
        
        prob = np.zeros_like(in_vals)
        
        idx = (in_vals > self.low_cutoff) & (in_vals < self.high_cutoff)
        
        prob[idx] = self.exp_noise.get_prob(in_vals[idx])

        prob[~idx] = self.gaussian_noise.get_prob(in_vals[~idx]) 
     
        return prob  
            
    def transform(self, in_vals):
        
        out_vals = np.zeros_like(in_vals)
                
        idx = (in_vals > self.low_cutoff) & (in_vals < self.high_cutoff)
                
        out_vals[idx] = self.exp_noise.transform(in_vals[idx])

        out_vals[~idx] = self.gaussian_noise.transform(in_vals[~idx]) 
     
        return out_vals  
    
        
class CompositeConditionalNoise:
    def __init__(self, empty_noise, nonempty_noise, empty_prob=0.5, cutoff_percent=0.95):
                
        self.empty_noise = empty_noise
        self.nonempty_noise = nonempty_noise
        self.empty_gaussian_noise = GaussianConditionalNoise(empty_noise.get_in_data(), empty_noise.get_out_data())
        self.nonempty_gaussian_noise = GaussianConditionalNoise(nonempty_noise.get_in_data(), nonempty_noise.get_out_data())
        
        
        self.empty_prob = empty_prob
        self.cutoff_percent = cutoff_percent
    
        self.low_cutoff = np.quantile(self.empty_noise.get_in_data(), 1.0-self.cutoff_percent)
#         self.low_cutoff = np.quantile(self.nonempty_noise.get_in_data(), 1.0-self.cutoff_percent)
        self.high_cutoff = np.quantile(self.nonempty_noise.get_in_data(), self.cutoff_percent)
    
    
    def plot(self, ax, color='b'):
        
        self.empty_noise.plot(ax, color='g', cbar=False)
        self.nonempty_noise.plot(ax, color='b', cbar=False)
        
        
        
        ax.vlines([self.low_cutoff, self.high_cutoff], ymin=10**-2, ymax=10**7, color='r', ls='dashed')

        add_ellipse(ax, self.empty_gaussian_noise.mean, self.empty_gaussian_noise.cov)
        add_ellipse(ax, self.nonempty_gaussian_noise.mean, self.nonempty_gaussian_noise.cov)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlim(10**-2, 10**6)
        ax.set_ylim(10**-1, 10**7)
        
    def plot_composite(self, ax, color='b'):
        
        empty_hist, xedges, yedges  = np.histogram2d(np.log10(self.empty_noise.get_in_data()), 
                                 np.log10(self.empty_noise.get_out_data()),
                                bins=100, range=[(np.log10(self.low_cutoff), np.log10(self.high_cutoff)), (-1, 7)], density=True)
        
        nonempty_hist, xedges, yedges  = np.histogram2d(np.log10(self.nonempty_noise.get_in_data()), 
                                                     np.log10(self.nonempty_noise.get_out_data()),
                                                    bins=100, range=[(np.log10(self.low_cutoff), np.log10(self.high_cutoff)), (-1, 7)], density=True)
        
 
        hist = self.empty_prob*empty_hist + (1-self.empty_prob)*nonempty_hist
       
        X, Y = np.meshgrid(xedges, yedges)
        
        norm = mcolors.Normalize(np.min(hist)/2, np.max(hist))
        cmap = plt.cm.Blues_r
        
        hist = ma.masked_equal(hist, 0)
        
        ax.pcolormesh(10**X, 10**Y, hist.T, cmap=cmap, norm=norm, rasterized=True, shading='auto')
        
        ax.vlines([self.low_cutoff, self.high_cutoff], ymin=10**-2, ymax=10**7, color='r', ls='dashed')

        add_ellipse(ax, self.empty_gaussian_noise.mean, self.empty_gaussian_noise.cov)
        add_ellipse(ax, self.nonempty_gaussian_noise.mean, self.nonempty_gaussian_noise.cov)
        
        
        x = np.linspace(np.log10(self.high_cutoff), 6, 50)
        y = np.linspace(-1, 7, 200)

        X, Y = np.meshgrid(x, y)
        Z = self.empty_prob*self.empty_gaussian_noise.get_joint_prob(10**X, 10**Y) + (1-self.empty_prob)*self.nonempty_gaussian_noise.get_joint_prob(10**X, 10**Y)
        
        Z = ma.masked_less(Z, np.min(hist)/2)
        ax.pcolormesh(10**X, 10**Y, Z, cmap=cmap, norm=norm, rasterized=True, shading='auto')
        
        
        x = np.linspace(-2, np.log10(self.low_cutoff), 50)
        y = np.linspace(-1, 7, 200)

        X, Y = np.meshgrid(x, y)
        Z = self.empty_prob*self.empty_gaussian_noise.get_joint_prob(10**X, 10**Y) + (1-self.empty_prob)*self.nonempty_gaussian_noise.get_joint_prob(10**X, 10**Y)
        
        Z = ma.masked_less(Z, np.min(hist)/2)
        
        ax.pcolormesh(10**X, 10**Y, Z, cmap=cmap, norm=norm, rasterized=True, shading='auto')

        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlim(10**-2, 10**6)
        ax.set_ylim(10**-1, 10**7)
        
    def plot_conditional_prob(self, ax, cbar=False):
        
        
        empty_hist, xedges, yedges  = np.histogram2d(np.log10(self.empty_noise.get_in_data()), 
                                 np.log10(self.empty_noise.get_out_data()),
                                bins=100, range=[(np.log10(self.low_cutoff), np.log10(self.high_cutoff)), (-1, 7)], density=False)
        
        nonempty_hist, xedges, yedges  = np.histogram2d(np.log10(self.nonempty_noise.get_in_data()), 
                                                     np.log10(self.nonempty_noise.get_out_data()),
                                                    bins=100, range=[(np.log10(self.low_cutoff), np.log10(self.high_cutoff)), (-1, 7)], density=False)
        
        
     
        prob_anti = self.empty_prob*np.sum(empty_hist, axis=1) + (1-self.empty_prob)*np.sum(nonempty_hist, axis=1)
   
        hist = (self.empty_prob*empty_hist + (1-self.empty_prob)*nonempty_hist) / np.maximum(prob_anti[:, np.newaxis], 1.0)
        
        hist = ma.masked_equal(hist, 0)
        
        
        X, Y = np.meshgrid(xedges, yedges)
        
        norm = mcolors.LogNorm(np.min(hist), np.max(hist))
        cmap = plt.cm.Blues_r
        
        hist = ma.masked_equal(hist, 0)
        
        ax.pcolormesh(10**X, 10**Y, hist.T, cmap=cmap, norm=norm, rasterized=True, shading='auto')
        
        ax.vlines([self.low_cutoff, self.high_cutoff], ymin=10**-2, ymax=10**7, color='r', ls='dashed')
        
        
        x = np.linspace(np.log10(self.high_cutoff), 6, 50)
        y = np.linspace(-1, 7, 200)

        X, Y = np.meshgrid(x, y)
        
        empty_proba = self.empty_gaussian_noise.get_prob(10**X)
        nonempty_proba = self.nonempty_gaussian_noise.get_prob(10**X)
        proba = self.empty_prob * empty_proba + (1-self.empty_prob)*nonempty_proba
        
        empty_prob = self.empty_gaussian_noise.get_conditional_prob(10**X, 10**Y) * self.empty_prob * empty_proba / proba
        nonempty_prob = self.nonempty_gaussian_noise.get_conditional_prob(10**X, 10**Y) * (1-self.empty_prob)*nonempty_proba / proba 
            
        Z = (empty_prob + nonempty_prob)[:-1, :-1]
        dy = y[1:]-y[:-1]
        Z = Z * dy[:, np.newaxis]
        Z = ma.masked_less(Z, np.min(hist))
        
        ax.pcolormesh(10**X, 10**Y, Z, cmap=cmap, norm=norm, rasterized=True, shading='auto')
        
        
        x = np.linspace(-2, np.log10(self.low_cutoff), 50)
        y = np.linspace(-1, 7, 200)

        X, Y = np.meshgrid(x, y)
        
        empty_proba = self.empty_gaussian_noise.get_prob(10**X)
        nonempty_proba = self.nonempty_gaussian_noise.get_prob(10**X)
        proba = self.empty_prob * empty_proba + (1-self.empty_prob)*nonempty_proba
        
        empty_prob = self.empty_gaussian_noise.get_conditional_prob(10**X, 10**Y) * self.empty_prob * empty_proba / proba
        nonempty_prob = self.nonempty_gaussian_noise.get_conditional_prob(10**X, 10**Y) * (1-self.empty_prob)*nonempty_proba / proba 
            
        Z = (empty_prob + nonempty_prob)[:-1, :-1]
        dy = y[1:]-y[:-1]
        Z = Z * dy[:, np.newaxis]
        Z = ma.masked_less(Z, np.min(hist))
        
        ax.pcolormesh(10**X, 10**Y, Z, cmap=cmap, norm=norm, rasterized=True, shading='auto')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlim(10**-2, 10**6)
        ax.set_ylim(10**-1, 10**7)
        
        
        if cbar:
            
            bbox = ax.get_position()

            cax = ax.figure.add_axes([bbox.x1+0.02, bbox.y0, 0.02, bbox.y1-bbox.y0])
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')

            cax.tick_params(which='major', direction='out', length=3.0, width=1.0)

            cbar.set_label(r"Conditional Frequency")
            
            
    def transform(self, in_vals):
        
        out_vals = np.zeros_like(in_vals)
                        
        # joint probability of anti and empty
        prob_a_and_empty = np.zeros_like(in_vals)
         
        # values that are within cutoffs
        idx = (in_vals > self.low_cutoff) & (in_vals < self.high_cutoff)
        
        # conditional probability of anti value given empty
        prob_a_given_empty = self.empty_noise.get_prob(in_vals[idx])
        # conditional probability of anti value given not empty
        prob_a_given_nonempty = self.nonempty_noise.get_prob(in_vals[idx])   
        # compute joint probability
        prob_a_and_empty[idx] = self.empty_prob*prob_a_given_empty / (self.empty_prob*prob_a_given_empty + (1-self.empty_prob)*prob_a_given_nonempty)

        out_vals[idx & (prob_a_and_empty > 0.0)] += prob_a_and_empty[idx & (prob_a_and_empty > 0.0)] * self.empty_noise.transform(in_vals[idx & (prob_a_and_empty > 0.0)])
        out_vals[idx & (prob_a_and_empty < 1.0)] += (1-prob_a_and_empty[idx & (prob_a_and_empty < 1.0)]) * self.nonempty_noise.transform(in_vals[idx & (prob_a_and_empty < 1.0)])
     
        # set values outside of cutoffs using gaussians
        prob_a_given_empty = self.empty_gaussian_noise.get_prob(in_vals[~idx])
        prob_a_given_nonempty = self.nonempty_gaussian_noise.get_prob(in_vals[~idx])        
        prob_a_and_empty[~idx] = self.empty_prob*prob_a_given_empty / (self.empty_prob*prob_a_given_empty + (1-self.empty_prob)*prob_a_given_nonempty)
         
        out_vals[~idx & (prob_a_and_empty > 0.0)] += prob_a_and_empty[~idx & (prob_a_and_empty > 0.0)] * self.empty_gaussian_noise.transform(in_vals[~idx & (prob_a_and_empty > 0.0)])
        out_vals[~idx & (prob_a_and_empty < 1.0)] += (1-prob_a_and_empty[~idx & (prob_a_and_empty < 1.0)]) * self.nonempty_gaussian_noise.transform(in_vals[~idx & (prob_a_and_empty < 1.0)]) 
                     
        return out_vals
    
    def get_prob_empty(self, in_vals):
            
        prob_a_and_empty = np.zeros_like(in_vals)
                
        idx = (in_vals > self.low_cutoff) & (in_vals < self.high_cutoff)
        
        prob_a_given_empty = self.empty_noise.get_prob(in_vals[idx])
        prob_a_given_nonempty = self.nonempty_noise.get_prob(in_vals[idx])        
        prob_a_and_empty[idx] = self.empty_prob*prob_a_given_empty / (self.empty_prob*prob_a_given_empty + (1-self.empty_prob)*prob_a_given_nonempty)

     
        prob_a_given_empty = self.empty_gaussian_noise.get_prob(in_vals[~idx])
        prob_a_given_nonempty = self.nonempty_gaussian_noise.get_prob(in_vals[~idx])        
        prob_a_and_empty[~idx] = self.empty_prob*prob_a_given_empty / (self.empty_prob*prob_a_given_empty + (1-self.empty_prob)*prob_a_given_nonempty)
         
        return prob_a_and_empty

    
    
    
class PercentileNoise:
    def __init__(self, in_data, out_data, nbins, verbose=False):
        
        self.in_data = in_data
        self.out_data = out_data
        
        self.nbins = nbins
        
        
        in_percentiles = np.percentile(in_data, np.linspace(0, 100, nbins+1))
        out_percentiles = np.percentile(out_data, np.linspace(0, 100, nbins+1))
        
        self.df = pd.DataFrame(np.c_[np.linspace(0, 100, nbins+1), in_percentiles, out_percentiles], columns=['percentile', 'in_data', 'out_data'])
        
        self.out_mean = np.sqrt(out_percentiles[0:nbins]*out_percentiles[1:nbins+1])
        self.in_mean = np.sqrt(in_percentiles[0:nbins]*in_percentiles[1:nbins+1])
        
        
    def plot(self, ax, color='b'):
        
        ax.scatter(self.df['in_data'], self.df['out_data'], s=1+self.df['percentile'])
        ax.plot(self.df['in_data'], self.df['out_data'], color='k')
        ax.scatter(self.in_mean, self.out_mean, s=20, color='r', marker='x')
        
    
    def get_in_data(self):
        return self.in_data
    
    def get_out_data(self):
        return self.out_data
    
    def get_bin_index(self, in_vals):
                            
        bin_index = np.digitize(in_vals, self.df['in_data'], right=False)-1

        bin_index[bin_index==self.nbins] = -1
        
                
        return bin_index
    
        
    def transform(self, in_vals):
        
        in_bin_index = self.get_bin_index(in_vals)
                
        out_vals = np.full_like(in_bin_index, -1.0, dtype=float)
        
        out_vals[in_bin_index==-1] = np.nan
        
        idx = in_bin_index!=-1
        out_vals[idx] = self.out_mean[in_bin_index[idx]]
                        
        return out_vals
    
    
    def get_prob(self, in_vals):
        
        in_bin_index = self.get_bin_index(in_vals)
        
        unique_bins = np.unique(in_bin_index)
                
        prob = np.zeros_like(in_vals)
        
        for b in unique_bins:
            
            idx = in_bin_index==b
            
            if b == -1:
                continue
                        
            prob[idx] = self.prob_in_vals[b]
                        
        return prob
    
    
    
    
# This function decomposes nonempty_noise into a guassian mixture of two components.
# The first component is simply empty_noise and the second is found by EM 
def gaussian_mixture(empty_noise, nonempty_noise, tol=1e-4):
        
    # Calculate mixture component for empty cells
    empty_mean = np.mean(np.log10(empty_noise.df[['in_data', 'out_data']].to_numpy()), axis=0)
    empty_cov = np.cov(np.log10(empty_noise.df[['in_data', 'out_data']].to_numpy()), rowvar=False)
    
    # Initial mixture component parameters for nonempty cells
    nonempty_mean = empty_mean.copy()
    nonempty_cov = empty_cov.copy()
    q_empty = 0.01
    
        
    def log_likelihood(x, y, mu, Sigma, q):
        
        return -np.sum(np.log((1-q)*prob_normal(x, y, mu, Sigma) + q*prob_normal(x, y, empty_mean, empty_cov)))/len(x)
        
    x = np.log10(nonempty_noise.df['in_data'])
    y = np.log10(nonempty_noise.df['out_data'])
    
    loss = log_likelihood(x, y, nonempty_mean, nonempty_cov, q_empty)
#     print("Initial Loss:", loss)
#     print(nonempty_mean)
#     print(nonempty_cov)
#     print(q_empty)
    
    n_max = 100
    n = 0
    while n < n_max:
        n += 1
        
        # expectation step
        # evaluate posterior probabilities for latent variables representing which mixture each data point is pulled from
        gamma_empty = q_empty * prob_normal(x, y, empty_mean, empty_cov)
        gamma_nonempty = (1-q_empty) * prob_normal(x, y, nonempty_mean, nonempty_cov)
        
        # normalize
        norm = gamma_empty + gamma_nonempty
        gamma_empty /= norm
        gamma_nonempty /= norm
        
        # effective number of data points from each mixture component
        N_empty = np.sum(gamma_empty)
        N_nonempty = np.sum(gamma_nonempty)
        
        # maximization step
        # estimate new parameters
        
        nonempty_mean_est = np.zeros_like(nonempty_mean)
        nonempty_mean_est[0] = 1/N_nonempty * np.sum(gamma_nonempty*x)
        nonempty_mean_est[1] = 1/N_nonempty * np.sum(gamma_nonempty*y)
        
        nonempty_cov_est = np.zeros_like(nonempty_cov)
        nonempty_cov_est[0, 0] = 1/N_nonempty * np.sum(gamma_nonempty*(x-nonempty_mean[0])**2)
        nonempty_cov_est[1, 1] = 1/N_nonempty * np.sum(gamma_nonempty*(y-nonempty_mean[1])**2)
        nonempty_cov_est[0, 1] = 1/N_nonempty * np.sum(gamma_nonempty*(x-nonempty_mean[0])*(y-nonempty_mean[1]))
        nonempty_cov_est[1, 0] = nonempty_cov_est[0, 1]
        
        q_empty_est = N_empty / len(x)
        
        loss_est = log_likelihood(x, y, nonempty_mean_est, nonempty_cov_est, q_empty_est)
        
#         print("Step: ", n, "Loss: ", loss_est)
        
        if loss - loss_est < tol:
            break
        else:
            loss = loss_est
            nonempty_mean = nonempty_mean_est
            nonempty_cov = nonempty_cov_est
            q_empty = q_empty_est
            
#             print(nonempty_mean)
#             print(nonempty_cov)
#             print(q_empty)

    print("Iters:", n, "/", n_max)
    print("q_empty", q_empty)
        
    return empty_mean, empty_cov, nonempty_mean, nonempty_cov, q_empty
        
    
class GaussianMixtureNoise:
    
    def __init__(self, empty_noise, nonempty_noise, verbose=False, tol=1e-4):
                
        self.empty_noise = empty_noise
        self.nonempty_noise = nonempty_noise
        
        self.empty_mean, self.empty_cov, self.nonempty_mean, self.nonempty_cov, self.q_empty = gaussian_mixture(empty_noise, nonempty_noise, tol=tol)
        
#         self.in_mean = np.mean(np.log10(self.get_in_data()))
#         self.out_mean = np.mean(np.log10(self.get_out_data()))
        
#         self.cov = np.cov(np.log10(self.df.to_numpy()), rowvar=False)

#         self.evals, self.evecs = la.eigh(self.cov)
        
#         self.line_vec = self.evecs[:, 0]
#         self.sigma = np.sqrt(self.evals[0])
    
    def plot(self, ax):
        
        x = np.linspace(-2, 7, 200)
        y = np.linspace(-2, 7, 200)

        X, Y = np.meshgrid(x, y)
        Z = self.joint_prob(X, Y)
        ax.pcolormesh(10**X, 10**Y, Z, cmap=plt.cm.Blues, rasterized=True, shading='auto')

        add_ellipse(ax, self.empty_mean, self.empty_cov)
        add_ellipse(ax, self.nonempty_mean, self.nonempty_cov)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        
    def plot_conditional_prob(self, ax, cbar=False):
        
        x = np.linspace(-2, 7, 200)
        y = np.linspace(-2, 7, 200)

        X, Y = np.meshgrid(x, y)
        Z = self.conditional_prob(X, Y)[:-1, :-1]
        dy = y[1:]-y[:-1]
        Z = Z * dy[:, np.newaxis]
        
        norm = mcolors.LogNorm(1e-6, 1.0)
#         norm = mcolors.Normalize(np.min(Z), np.max(Z))
        cmap=plt.cm.Blues

        ax.pcolormesh(10**X, 10**Y, Z, 
                      cmap=cmap, norm=norm,
                      rasterized=True, shading='auto')
        
        add_ellipse(ax, self.empty_mean, self.empty_cov)
        add_ellipse(ax, self.nonempty_mean, self.nonempty_cov)

        ax.set_xscale('log')
        ax.set_yscale('log')

        
        if cbar:
            
            bbox = ax.get_position()

            cax = ax.figure.add_axes([bbox.x1+0.2, bbox.y0, 0.02, bbox.y1-bbox.y0])
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')

            cax.tick_params(which='major', direction='out', length=3.0, width=1.0)

            cbar.set_label(r"Conditional Frequency")
        
    def joint_prob(self, x, y):
        
        return self.q_empty*prob_normal(x, y, self.empty_mean, self.empty_cov) + (1-self.q_empty)*prob_normal(x, y, self.nonempty_mean, self.nonempty_cov)
        
        
    def conditional_prob(self, x, y, q=None):
                
        if q is None:
            q = self.q_empty
            
        Sigmainv = la.inv(self.empty_cov)
        detSigma = la.det(self.empty_cov)
        
        dx = x - self.empty_mean[0]
        dy = y - self.empty_mean[1]
        
        joint_prob_empty = np.exp(-(Sigmainv[0, 0]*dx**2+Sigmainv[1, 1]*dy**2+2*Sigmainv[0, 1]*dx*dy)/2.0) / np.sqrt((2*np.pi)**2*detSigma)
        
        probx_empty = np.exp(-(Sigmainv[0, 0] - Sigmainv[0, 1]**2/Sigmainv[1, 1])*dx**2/2.0) / np.sqrt((2*np.pi)**2*detSigma) * np.sqrt(2*np.pi/Sigmainv[1, 1]**2)
        
        Sigmainv = la.inv(self.nonempty_cov)
        detSigma = la.det(self.nonempty_cov)
        
        dx = x - self.nonempty_mean[0]
        dy = y - self.nonempty_mean[1]
        
        joint_prob_nonempty = np.exp(-(Sigmainv[0, 0]*dx**2+Sigmainv[1, 1]*dy**2+2*Sigmainv[0, 1]*dx*dy)/2.0) / np.sqrt((2*np.pi)**2*detSigma)
        
        probx_nonempty = np.exp(-(Sigmainv[0, 0] - Sigmainv[0, 1]**2/Sigmainv[1, 1])*dx**2/2.0) / np.sqrt((2*np.pi)**2*detSigma) * np.sqrt(2*np.pi/Sigmainv[1, 1]**2)

        return (q*joint_prob_empty + (1-q)*joint_prob_nonempty) / (q*probx_empty + (1-q)*probx_nonempty)
        
        
        
#     def transform(self, in_vals):
        
#         x = (np.log10(in_vals) - self.in_mean) * self.line_vec[0]/self.line_vec[1]
        
#         out_vals = 10**rand.normal(loc=self.out_mean-x, scale=self.sigma/self.line_vec[1])
                        
#         return out_vals
    
    
  