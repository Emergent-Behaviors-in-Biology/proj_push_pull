import numpy as np
import scipy as sp
import numpy.ma as ma
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.patches as mpatches
import matplotlib.collections as mcollect


"""
Description:

Creates a 2d histogram, but then plots the average value of a separate quantity in each bin.

Arguments:

df: Dataframe containing data that you wish to plot.
fig: matplotlib figure
ax: set of axes within fig where the plot should be placed
xlabel: column of df that should be plotted along the x-axis
ylabel: column of df that should be plotted along the y-axis
zlabel: column of df that that will be used to color each bin
vmin: lower bound for coloring each pixel
vmax: upper bound for coloring each pixel
logscale: whether to apply log scaling pixel coloring
xlim: limits of x-axis
ylim: limits of y-axis
"""

def plot_2d_avg(df, fig, ax, xlabel, ylabel, zlabel, vmin=1e0, vmax=1e2, logscale=True, xlim=(1e1, 1e5), ylim=(1e1, 1e5)):
    
    
    # creates 2d histogram in log space
    hist, xedges = np.histogram(np.log10(df[xlabel]), bins='auto')
    hist, yedges = np.histogram(np.log10(df[ylabel]), bins='auto')
    # converts bin edges back to linear scale
    xedges = 10**xedges
    yedges = 10**yedges

    # sort data into bins found by histogram
    df['xbin'] = pd.cut(df[xlabel], xedges, labels=False)
    df['ybin'] = pd.cut(df[ylabel], yedges, labels=False)

    # calculate mean of zlabel values in each bin
    hist_corr = df.groupby(['xbin', 'ybin'])[zlabel].mean().to_frame("mean")

    # add nan values to bins that do not have any data points
    hist_corr =  hist_corr.reindex(pd.MultiIndex.from_product([np.arange(len(xedges)-1), np.arange(len(yedges)-1)], 
                                                names=['xbin', 'ybin']), fill_value=np.nan)

    # pivot the data into a 2d grid
    hist_corr = hist_corr.reset_index().pivot(index='xbin', columns='ybin', values='mean').values

    # mask bins with no data
    hist_corr = ma.masked_invalid(hist_corr)

    # create matplotlib colorbar properties
    if logscale:
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    cmap=plt.cm.viridis

    # plot image
    X, Y = np.meshgrid(xedges, yedges)
    im = ax.pcolormesh(X, Y, hist_corr.T, cmap=cmap, norm=norm, rasterized=True)

    t = np.linspace(*xlim)
    ax.plot(t, t, 'k--')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # ensure aspect ratio is square
    ax.set_aspect(1 / ax.get_data_ratio())

    # place colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', aspect=15)
    cbar.ax.tick_params(which='major', direction='out', length=3.0, width=1.0)
    cbar.set_label(zlabel)
    
    
"""
Description:

Creates a 2d histogram with hexagonal bins, but then plots the average value of a separate quantity in each bin.

Arguments:

df: Dataframe containing data that you wish to plot.
fig: matplotlib figure
ax: set of axes within fig where the plot should be placed
xlabel: column of df that should be plotted along the x-axis
ylabel: column of df that should be plotted along the y-axis
zlabel: column of df that that will be used to color each bin
nbins: number of bins in x direction. These bins are defined with respect to the the full x limits. Number of bins in y direction is chosen automatically.
vmin: lower bound for coloring each pixel
vmax: upper bound for coloring each pixel
xlim: limits of x-axis
ylim: limits of y-axis
"""

def plot_2d_avg_hex(df, fig, ax, xlabel, ylabel, zlabel, nbins=20, vmin=None, vmax=None, xlim=(1e1, 1e5), ylim=(1e1, 1e5)):
    
    
    # number of points in each bin
    hex_count = ax.hexbin(df[xlabel], df[ylabel], gridsize=nbins, 
                          extent=(np.log10(xlim[0]), np.log10(xlim[1]), np.log10(ylim[0]), np.log10(ylim[1])), 
                          xscale='log', yscale='log', mincnt=1)
    hex_count.remove()
    
    counts = hex_count.get_array()
    count_norm = mpl.colors.LogNorm(vmin=1.0, vmax=np.max(counts), clip=True)
        
    x = []
    y = []
    
    patches = []
    for i, path in enumerate(hex_count.get_paths()):

        # positions of vertices for hexagon
        verts = path.vertices
        
        # center of hexagon
        center = np.mean(np.log10(path.vertices[:-1]), axis=0)
        
        # size of hexagon (ranges from 0.0 to 1.0)
        scale = count_norm(counts[i])
#         scale = 1.0
        
        # create new hexagon
        patch = mpatches.Polygon(10**(scale*(np.log10(verts) - center)+center), closed=True)
        
        patches.append(patch)
        
    # calculate average value in each bin
    hex_val = ax.hexbin(df[xlabel], df[ylabel], C=df[zlabel], gridsize=nbins, 
                        extent=(np.log10(xlim[0]), np.log10(xlim[1]), np.log10(ylim[0]), np.log10(ylim[1])),
                        xscale='log', yscale='log', reduce_C_function=np.mean)
    hex_val.remove()
    
    values = hex_val.get_array()
    
    if vmin == None:
        vmin = values.min()
    if vmax == None:
        vmax = values.max()
    val_norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap=plt.cm.viridis
    
    pc = mcollect.PatchCollection(patches, linewidths=1, norm=val_norm, cmap=cmap)
    pc.set_array(values)
    
    ax.add_collection(pc)
        
    t = np.linspace(*xlim)
    ax.plot(t, t, 'k--')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # ensure aspect ratio is square
    ax.set_aspect(1 / ax.get_data_ratio())

    # plot colorbar
    # make fake image first
    im = ax.pcolormesh(np.array([[0, 1]]), cmap=cmap, norm=val_norm)
    im.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', aspect=15)
    cbar.ax.tick_params(which='major', direction='out', length=3.0, width=1.0)
    cbar.set_label(zlabel)
    
    

    
    
"""
Description:

Sorts data into groups according to amount of substrate and then plots the data belonging to each group as a separate curve.

Arguments:

df: Dataframe containing data that you wish to plot.
fig: matplotlib figure
ax: set of axes within fig where the plot should be placed
WT_label: column of df containing writer concentration
ST-label: column of df containing substrate concentration
SpT_label: column of df containing phosphorylated substrate concentration
fmt: matplotlib format used for each curve
normalizex: whether to normalized writer by total amount of substrate
normalizex: whether to normalized phosphorylated substrate by total amount of substrate
xlim: limits of x-axis
ylim: limits of y-axis
"""

def plot_activation_curves(df, fig, ax, WT_label, ST_label, SpT_label, fmt='.--', normalizex=False, normalizey=False, xlim=(1e2, 1e5), ylim=(1e2, 1e4)):
    
    
    # sort data into bins according to ST_label
    bins, ST_edges = pd.qcut(df[ST_label], 10, labels=False, retbins=True)
    
    df['ST_bin'] = bins
      
    # calculate average value of ST in each bin
    ST_avg = df.groupby(['ST_bin'])[ST_label].mean()
      
    # create smap to color each curve according to ST
    norm = mpl.colors.LogNorm(vmin=min(ST_avg.min(), 1e2), vmax=max(ST_avg.max(), 1e4))
    cmap=plt.cm.cividis
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
     
    # iterate through each bin and plot activtion curve
    for ibin, STgroup in df.groupby(['ST_bin']):

        if normalizex:
            STgroup['x'] = STgroup[WT_label]  / ST_avg[ibin]  
        else:
            STgroup['x'] = STgroup[WT_label]
            
        if normalizey:
            STgroup['y'] = STgroup[SpT_label]  / ST_avg[ibin]            
        else:
            STgroup['y'] = STgroup[SpT_label]
        
        
        # sort data within bin into new bins along x axis
        bins, edges = pd.qcut(STgroup['x'], 10, 
                                    labels=False, retbins=True)
        STgroup['xbin'] = bins
        
        # calculate mean value of points in each bin
        x = STgroup.groupby('xbin')['x'].mean() 
        y = STgroup.groupby('xbin')['y'].mean() 
        yerr =  STgroup.groupby('xbin')['y'].sem()

        # plot activation curve
        ax.errorbar(x, y, yerr=yerr, 
                    fmt='.-', label=r"$S_T={0:.2f}$".format(ST_avg[ibin]), 
                    color=smap.to_rgba(ST_avg[ibin]), ms=6.0, lw=1.5)

    
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    if normalizey:
        ax.set_ylabel("Fraction Phospho Substrate")
    else:
        ax.set_yscale('log')
        ax.set_ylabel("Phospho Substrate")
    

    if normalizex:
        ax.set_xlabel("Writer/Substrate")
    else:
        ax.set_xlabel("Writer")
    
    ax.set_xscale('log')
    
    
    # plot colorbar
    # make fake image first
    im = ax.pcolormesh(np.array([[0, 1]]), cmap=cmap, norm=norm)
    im.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', aspect=15)
    cbar.ax.tick_params(which='major', direction='out', length=3.0, width=1.0)
    cbar.set_label(ST_label)
    
    
    

    
def plot_activation_theory(df, fig, ax, WT_label, ST_label, SpT_label, normalizex=False, normalizey=False):
    
    
    bins, ST_edges = pd.qcut(df[ST_label], 10, labels=False, retbins=True)
    
    df['ST_bin'] = bins
            
    ST_avg = df.groupby(['ST_bin'])[ST_label].mean()
      
        
    norm = mpl.colors.LogNorm(vmin=min(ST_avg.min(), 1e2), vmax=max(ST_avg.max(), 1e4))
    cmap=plt.cm.cividis
    
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        
    for ibin, STgroup in df.groupby(['ST_bin']):
        
        if ibin == 0:
            continue
        
        if normalizex:
            STgroup['x'] = STgroup[WT_label]  / ST_avg[ibin]
        else:
            STgroup['x'] = STgroup[WT_label]
            
        if normalizey:
            STgroup['y'] = STgroup[SpT_label]  / ST_avg[ibin]
        else:
            STgroup['y'] = STgroup[SpT_label]
        
        
        bins, edges = pd.qcut(STgroup['x'], 10, 
                                    labels=False, retbins=True)
        
        STgroup['xbin'] = bins
        
        x = STgroup.groupby('xbin')['x'].median() 
        y = STgroup.groupby('xbin')['y'].median() 
#         Sp_frac_err_up = STgroup.groupby('ratio_bin')['SpT_ST_ratio'].quantile(0.75)-Sp_frac
#         Sp_frac_err_low = Sp_frac-STgroup.groupby('ratio_bin')['SpT_ST_ratio'].quantile(0.25)
        yerr =  STgroup.groupby('xbin')['y'].sem()

        ax.plot(x, y, '--', color=smap.to_rgba(ST_avg[ibin]), ms=6.0, lw=1.0)

#         ax.plot(WT_ST_ratio, Sp_frac, fmt, label=r"$S_T={0:.2f}$".format(ST_avg[ibin]), 
#                     color=smap.to_rgba(ST_avg[ibin]), ms=6.0)
        
    