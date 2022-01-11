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
xlabel, ylabel, zlable: columns of df that should be plotted along the x-axis, y-axis
zlabel: column of df that that will be used to color each bin
nbins: number of bins in x direction. These bins are defined with respect to the the full x limits. Number of bins in y direction is chosen automatically.
vmin: lower bound for coloring each pixel
vmax: upper bound for coloring each pixel
xlim: limits of x-axis
ylim: limits of y-axis
logscale: whether to color pixels using log scale
normalize_xlabel, normalize_ylabel, normalize_zlabel: columns of df to normalize x, y, and z data by
show_diagonal: whether to show dashed line along diagonal of plot
"""

def plot_2d_avg_hex(df, fig, ax, xlabel, ylabel, zlabel, nbins=20, vmin=None, vmax=None, xlim=(1e1, 1e5), ylim=(1e1, 1e5), logscale=False,
                   normalize_xlabel=None, normalize_ylabel=None, normalize_zlabel=None, show_diagonal=False):
    
    x = df[xlabel].values
    y = df[ylabel].values
    z = df[zlabel].values
    
    if normalize_xlabel is not None:
        x = x / df[normalize_xlabel].values
        
    if normalize_ylabel is not None:
        y = y / df[normalize_ylabel].values
        
    if normalize_zlabel is not None:
        z = z / df[normalize_zlabel].values
        
    
    # number of points in each bin
    hex_count = ax.hexbin(x, y, gridsize=nbins, 
                          extent=(np.log10(xlim[0]), np.log10(xlim[1]), np.log10(ylim[0]), np.log10(ylim[1])), 
                          xscale='log', yscale='log', mincnt=1)
    hex_count.remove()
    
    counts = hex_count.get_array()
    count_norm = mpl.colors.LogNorm(vmin=1.0, vmax=np.max(counts), clip=True)
        
    patches = []
    for i, path in enumerate(hex_count.get_paths()):

        # positions of vertices for hexagon
        verts = path.vertices
        
        # center of hexagon
        center = np.mean(np.log10(path.vertices[:-1]), axis=0)
        
        # size of hexagon (ranges from 0.0 to 1.0)
        scale = count_norm(counts[i])
#         scale = 1.0
        if scale < 0.3:
            scale = 0.0
        
        # create new hexagon
        patch = mpatches.Polygon(10**(scale*(np.log10(verts) - center)+center), closed=True)
        
        patches.append(patch)
        
    # calculate average value in each bin
    hex_val = ax.hexbin(x, y, C=z, gridsize=nbins, 
                        extent=(np.log10(xlim[0]), np.log10(xlim[1]), np.log10(ylim[0]), np.log10(ylim[1])),
                        xscale='log', yscale='log', reduce_C_function=np.mean)
    hex_val.remove()
    
    values = hex_val.get_array()
    
    if vmin == None:
        vmin = values.min()
    if vmax == None:
        vmax = values.max()
        
    if logscale:
        val_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        val_norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap=plt.cm.viridis
    
    pc = mcollect.PatchCollection(patches, linewidths=1, norm=val_norm, cmap=cmap)
    pc.set_array(values)
    
    ax.add_collection(pc)
        
    if show_diagonal:
        t = np.linspace(*xlim)
        ax.plot(t, t, 'k--')

    ax.set_xscale('log')
    ax.set_yscale('log')

    if normalize_xlabel is None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel + "/" + normalize_xlabel)
        
    if normalize_ylabel is None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(ylabel + "/" + normalize_ylabel)
    
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
    
    if normalize_zlabel is None:
        cbar.set_label(zlabel)
    else:
        cbar.set_label(zlabel + "/" + normalize_zlabel)
        
    

    
    
"""
Description:

Sorts data into groups according to amount of substrate and then plots the data belonging to each group as a separate curve.

Arguments:

df: Dataframe containing data that you wish to plot.
fig: matplotlib figure
ax: set of axes within fig where the plot should be placed
xlabel, ylabel: columns of df to plot on x-axis and y-axis
zlabel: column of df that is held constant for each curve, data will be sorted into groups along this column
fmt: matplotlib format used for each curve
normalize_xlabel, normalize_ylabel, normalize_zlabel: columns of df to normalize x, y, and z data by
xlim: limits of x-axis
ylim: limits of y-axis
nxbins: number of bins along the x-axis for each curve
nSTbins: number of bins along ST_label
error_bands: whether to display error bands
use_median: whether to display median value in each bin or mean value
error_band_range: upper and lower quantiles for each error band
xlog_scale, ylog_scale: wether to plot x and y axes using log scale
"""

def plot_activation_curves(df, fig, ax, xlabel, ylabel, zlabel, fmt='.--', 
                           normalize_xlabel=None, normalize_ylabel=None, normalize_zlabel=None, xlim=(1e2, 1e5), ylim=(1e2, 1e4),
                            nxbins=10, nSTbins=10, error_bands=False, use_median=False, error_band_range=(0.5, 0.95),
                          xlog_scale=True, ylog_scale=True):
    
    
    x = df[xlabel].values
    y = df[ylabel].values
    z = df[zlabel].values
    
    if normalize_xlabel is not None:
        x = x / df[normalize_xlabel].values
        
    if normalize_ylabel is not None:
        y = y / df[normalize_ylabel].values
        
    if normalize_zlabel is not None:
        z = z / df[normalize_zlabel].values
        
    df_tmp = pd.DataFrame({'x': x, 'y': y, 'z': z})
    
    # sort data into bins according to ST_label
    zbins, z_edges = pd.qcut(z, nSTbins, labels=False, retbins=True)
    
    df_tmp['zbin'] = zbins
      
    # calculate average value of z in each bin
    z_avg = df_tmp.groupby(['zbin'])['z'].mean()
      
    # create smap to color each curve according to ST
    norm = mpl.colors.LogNorm(vmin=min(z_avg.min(), 1e2), vmax=max(z_avg.max(), 1e4))
    cmap=plt.cm.cividis
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
     
    # iterate through each bin and plot activation curve
    for ibin, zgroup in df_tmp.groupby(['zbin']):
        
        # sort data within bin into new bins along x axis
        xbins, edges = pd.qcut(zgroup['x'], nxbins, 
                                    labels=False, retbins=True)
        zgroup['xbin'] = xbins
        
        
        
        # calculate statistics of each bin
        if use_median:
            x = zgroup.groupby('xbin')['x'].median() 
            y = zgroup.groupby('xbin')['y'].median() 
        else:
            x = zgroup.groupby('xbin')['x'].mean() 
            y = zgroup.groupby('xbin')['y'].mean() 
            
        ax.plot(x, y, '-', color=smap.to_rgba(z_avg[ibin]))

            
        if error_bands:

            y_up = zgroup.groupby('xbin')['y'].quantile(0.95) 
            y_low = zgroup.groupby('xbin')['y'].quantile(0.05) 

            ax.fill_between(x=x, y1=y_low, y2=y_up, alpha=0.2, color=smap.to_rgba(z_avg[ibin]))

        

    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    
    if normalize_xlabel is None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel + "/" + normalize_xlabel)
        
    if normalize_ylabel is None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(ylabel + "/" + normalize_ylabel)
    
    if xlog_scale:
        ax.set_xscale('log')
        
    if ylog_scale:
        ax.set_yscale('log')
    
    # ensure aspect ratio is square
    ax.set_aspect(1 / ax.get_data_ratio())
    
    
    # plot colorbar
    # make fake image first
    im = ax.pcolormesh(np.array([[0, 1]]), cmap=cmap, norm=norm)
    im.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', aspect=15)
    cbar.ax.tick_params(which='major', direction='out', length=3.0, width=1.0)
    
    if normalize_zlabel is None:
        cbar.set_label(zlabel)
    else:
        cbar.set_label(zlabel + "/" + normalize_zlabel)
    
    
    
    
def plot_push_dataset_summary(df_data, dataset):
    
    df_tmp = df_data.query("dataset==@dataset")
    
    fig, axes = plt.subplots(2,2, constrained_layout=True, figsize=(8, 10))
    
    ##########################################################

    
    ax = axes[0, 0]
    
    plot_2d_avg_hex(df_tmp, fig, ax, 'WT_anti_exp', 'ST_anti_exp', 'SpT_anti_exp', nbins=20, xlim=(1e1, 1e5), ylim=(1e1, 1e5), show_diagonal=True)

    

    ##########################################################
    
    
    ax = axes[0, 1]
    
    plot_2d_avg_hex(df_tmp, fig, ax, 'WT_anti_exp', 'ST_anti_exp', 'SpT_anti_exp', nbins=20, xlim=(1e1, 1e5), ylim=(1e1, 1e5), 
                          vmin=0, vmax=1.5, logscale=True, normalize_zlabel='ST_anti_exp', show_diagonal=True)

    

    ##########################################################
    
    ax = axes[1, 0]
    
    plot_activation_curves(df_tmp, fig, ax, 'WT_anti_exp', 'SpT_anti_exp', 'ST_anti_exp', 
                             nSTbins=4, xlim=(1e1, 1e5), ylim=(1e1, 1e4), error_bands=True, use_median=True, error_band_range=(0.5, 0.95))


    ##########################################################
    
    ax = axes[1, 1]

    plot_activation_curves(df_tmp, fig, ax, 'WT_anti_exp', 'SpT_anti_exp', 'ST_anti_exp', 
                           normalize_xlabel='ST_anti_exp', normalize_ylabel='ST_anti_exp', xlim=(1e-1, 1e2), ylim=(0, 1.5), nSTbins=4,
                            error_bands=True, use_median=True, error_band_range=(0.5, 0.95), ylog_scale=False)
    
    ##########################################################

    fig.suptitle(dataset)

    plt.show()
    
    
def plot_pushpull_dataset_summary(df_data, dataset):
    
    df_tmp = df_data.query("dataset==@dataset")
    
    fig, axes = plt.subplots(2,6, constrained_layout=True, figsize=(24, 10))
    
    ##########################################################

    
    ax = axes[0, 0]
    
    plot_2d_avg_hex(df_tmp, fig, ax, 'WT_anti_exp', 'ST_anti_exp', 'SpT_anti_exp', nbins=20, xlim=(1e1, 1e5), ylim=(1e1, 1e5), show_diagonal=True)

    

    ##########################################################
    
    
    ax = axes[0, 1]
    
    plot_2d_avg_hex(df_tmp, fig, ax, 'WT_anti_exp', 'ST_anti_exp', 'SpT_anti_exp', nbins=20, xlim=(1e1, 1e5), ylim=(1e1, 1e5), 
                          vmin=0, vmax=1.5, logscale=True, normalize_zlabel='ST_anti_exp', show_diagonal=True)
    
    ##########################################################

    
    ax = axes[0, 2]
    
    plot_2d_avg_hex(df_tmp, fig, ax, 'ET_anti_exp', 'ST_anti_exp', 'SpT_anti_exp', nbins=20, xlim=(1e1, 1e5), ylim=(1e1, 1e5), show_diagonal=True)

    

    ##########################################################
    
    
    ax = axes[0, 3]
    
    plot_2d_avg_hex(df_tmp, fig, ax, 'ET_anti_exp', 'ST_anti_exp', 'SpT_anti_exp', nbins=20, xlim=(1e1, 1e5), ylim=(1e1, 1e5), 
                          vmin=0, vmax=1.5, logscale=True, normalize_zlabel='ST_anti_exp', show_diagonal=True)
    
    
    ##########################################################

    
    ax = axes[0, 4]
    
    plot_2d_avg_hex(df_tmp, fig, ax, 'WT_anti_exp', 'ST_anti_exp', 'SpT_anti_exp', nbins=20, xlim=(1e-3, 1e2), ylim=(1e1, 1e5),
                   normalize_xlabel='ET_anti_exp')

    

    ##########################################################
    
    
    ax = axes[0, 5]
    
    plot_2d_avg_hex(df_tmp, fig, ax, 'WT_anti_exp', 'ST_anti_exp', 'SpT_anti_exp', nbins=20, xlim=(1e-3, 1e2), ylim=(1e1, 1e5), 
                          vmin=0, vmax=1.5, logscale=True,  normalize_xlabel='ET_anti_exp', normalize_zlabel='ST_anti_exp')
    


    ##########################################################
    
    ax = axes[1, 0]
    
    plot_activation_curves(df_tmp, fig, ax, 'WT_anti_exp', 'SpT_anti_exp', 'ST_anti_exp', 
                             nSTbins=4, xlim=(1e1, 1e5), ylim=(1e1, 1e4), error_bands=True, use_median=True, error_band_range=(0.5, 0.95))


    ##########################################################
    
    ax = axes[1, 1]

    plot_activation_curves(df_tmp, fig, ax, 'WT_anti_exp', 'SpT_anti_exp', 'ST_anti_exp', 
                           normalize_xlabel='ST_anti_exp', normalize_ylabel='ST_anti_exp', xlim=(1e-1, 1e2), ylim=(0, 1.5), nSTbins=4,
                            error_bands=True, use_median=True, error_band_range=(0.5, 0.95), ylog_scale=False)
    
    
    
    ##########################################################
    
    ax = axes[1, 2]
    
    plot_activation_curves(df_tmp, fig, ax, 'ET_anti_exp', 'SpT_anti_exp', 'ST_anti_exp', 
                             nSTbins=4, xlim=(1e1, 1e5), ylim=(1e1, 1e4), error_bands=True, use_median=True, error_band_range=(0.5, 0.95))


    ##########################################################
    
    ax = axes[1, 3]

    plot_activation_curves(df_tmp, fig, ax, 'ET_anti_exp', 'SpT_anti_exp', 'ST_anti_exp', 
                           normalize_xlabel='ST_anti_exp', normalize_ylabel='ST_anti_exp', xlim=(1e-1, 1e2), ylim=(0, 1.5), nSTbins=4,
                            error_bands=True, use_median=True, error_band_range=(0.5, 0.95), ylog_scale=False)
    

    ##########################################################
    
    ax = axes[1, 4]
    
    plot_activation_curves(df_tmp, fig, ax, 'WT_anti_exp', 'SpT_anti_exp', 'ST_anti_exp', normalize_xlabel='ET_anti_exp',
                             nSTbins=4, xlim=(1e-2, 1e2), ylim=(1e1, 1e4), error_bands=True, use_median=True, error_band_range=(0.5, 0.95))


    ##########################################################
    
    ax = axes[1, 5]

    plot_activation_curves(df_tmp, fig, ax, 'WT_anti_exp', 'SpT_anti_exp', 'ST_anti_exp', 
                           normalize_xlabel='ET_anti_exp', normalize_ylabel='ST_anti_exp', xlim=(1e-2, 1e2), ylim=(0, 1.5), nSTbins=4,
                            error_bands=True, use_median=True, error_band_range=(0.5, 0.95), ylog_scale=False)
    
    

    ##########################################################

    fig.suptitle(dataset)

    plt.show()
    
    