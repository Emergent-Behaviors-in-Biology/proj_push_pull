import numpy as np
import scipy as sp
import numpy.ma as ma
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def plot_2d_avg(df, fig, ax, xlabel, ylabel, zlabel, vmax=None):
    
    hist, xedges = np.histogram(np.log10(df[xlabel]), bins='auto')
    hist, yedges = np.histogram(np.log10(df[ylabel]), bins='auto')
    xedges = 10**xedges
    yedges = 10**yedges

    df['xbin'] = pd.cut(df[xlabel], xedges, labels=False)
    df['ybin'] = pd.cut(df[ylabel], yedges, labels=False)


    hist_corr = df.groupby(['xbin', 'ybin'])[zlabel].median().to_frame("mean")

    hist_corr =  hist_corr.reindex(pd.MultiIndex.from_product([np.arange(len(xedges)-1), np.arange(len(yedges)-1)], 
                                                names=['xbin', 'ybin']), fill_value=np.nan)

    hist_corr = hist_corr.reset_index().pivot(index='xbin', columns='ybin', values='mean').values

    hist_corr = ma.masked_invalid(hist_corr)

    if vmax is None:
        vmax = df[zlabel].quantile(0.95)
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cmap=plt.cm.viridis

    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, hist_corr.T, cmap=cmap, norm=norm, rasterized=True)

    t = np.linspace(1e1, 1e5)
    ax.plot(t, t, 'k--')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_aspect(1 / ax.get_data_ratio())

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', aspect=15)
    cbar.ax.tick_params(which='major', direction='out', length=3.0, width=1.0)
    cbar.set_label(zlabel)
    
    
    
def plot_activation_curves(df, fig, ax, WT_label, ST_label, SpT_label, fmt='.--'):
    
    
    bins, ST_edges = pd.qcut(df[ST_label], 8, labels=False, retbins=True)
    
    df['ST_bin'] = bins
            
    ST_avg = df.groupby(['ST_bin'])[ST_label].mean()
      
        
    norm = mpl.colors.LogNorm(vmin=min(ST_avg.min(), 1e2), vmax=max(ST_avg.max(), 1e4))
    cmap=plt.cm.viridis
    
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        
    for ibin, STgroup in df.groupby(['ST_bin']):
        
        STgroup['WT_ST_ratio'] = STgroup[WT_label]  / ST_avg[ibin]
        
        STgroup['SpT_ST_ratio'] = STgroup[SpT_label]  / ST_avg[ibin]
        
        bins, ratio_edges = pd.qcut(STgroup['WT_ST_ratio'], 8, 
                                    labels=False, retbins=True)
        
        STgroup['ratio_bin'] = bins
        
        WT_ST_ratio = STgroup.groupby('ratio_bin')['WT_ST_ratio'].median() 
        Sp_frac = STgroup.groupby('ratio_bin')['SpT_ST_ratio'].median() 
        Sp_frac_err_up = STgroup.groupby('ratio_bin')['SpT_ST_ratio'].quantile(0.75)-Sp_frac
        Sp_frac_err_low = Sp_frac-STgroup.groupby('ratio_bin')['SpT_ST_ratio'].quantile(0.25)
            

#         ax.errorbar(WT_ST_ratio, Sp_frac, yerr=[Sp_frac_err_low, Sp_frac_err_up], 
#                     fmt='.--', label=r"$S_T={0:.2f}$".format(ST_avg[ibin]), 
#                     color=smap.to_rgba(ST_avg[ibin]), ms=6.0)

        ax.plot(WT_ST_ratio, Sp_frac, fmt, label=r"$S_T={0:.2f}$".format(ST_avg[ibin]), 
                    color=smap.to_rgba(ST_avg[ibin]), ms=6.0)
        
    
    
    ax.set_xscale('log')
    
#     ax.set_ylim(0, df[Sp_frac_label].quantile(0.95) / ST_avg[ibin])

    ax.set_ylabel("SpT/ST")
    ax.set_xlabel("WT/ST")
    
    im = ax.pcolormesh(np.array([[0, 1]]), cmap=cmap, norm=norm)
    im.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', aspect=2*15)
    cbar.ax.tick_params(which='major', direction='out', length=3.0, width=1.0)
    cbar.set_label(ST_label)