# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:09:21 2026

@author: wilfl

This file contains the code for creating and graphing statistics relating to the ejected droplets.
DataFrames should consist of columns named 'time','droplet_ID','vol_t','x','y','z','v_x','v_y','v_z'
The DataFrames should also have row indices reset such that the nth line has index n-1.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.colors as clr
import matplotlib.cm as cm

def extract_timestep(df,time):
    """
    Finds data from specified timestep
    
    Inputs
    -------
    df: pd.DataFrame
        Full dataset
    time: float
        Desired timestep
        
    Returns
    -------
    new_df: pd.DataFrame
        Subset of original DataFrame containing only values from the given timestep
    """
    new_df = df[abs(df['time']-time) < 1e-4].reset_index(drop=True)
    return new_df

def plot_droplet_count(df,dt,ax):
    """
    Plots the number of droplets over time
    
    Inputs
    -------
    df: pd.DataFrame
        Full dataset
    dt: float
        Length of timestep
    ax: mpl.axes._axes.Axes
        Axes to plot graph on
    """
    # Find start and end times
    end_time = df['time'][len(df)-1]
    time = df['time'][0]
    # Extract first timestep
    df_temp = extract_timestep(df,time)
    # Only take droplets smaller than original
    df_temp = df_temp[df_temp['vol_t']<2.0]
    # Create lists of timesteps and droplet counts
    times = [df['time'][0]]
    num_droplets = [len(df_temp)]
    time += dt
    # Loop through timesteps and append the times and counts to relevant lists
    while time <= end_time+0.5*dt:
        times.append(time)
        df_temp = extract_timestep(df,time)
        df_temp = df_temp[df_temp['vol_t']<2.0]
        num_droplets.append(len(df_temp))
        time += dt
    # Plot
    ax.plot(times, num_droplets, label='Number of Droplets')
    
def average_speed(df):
    """
    Finds average speed of droplets from a dataset
    
    Inputs
    -------
    df: pd.DataFrame
        Dataset to find average droplet speed from
        
    Returns
    -------
    average_speed: float
        Average speed of droplets from dataset
    """
    total_speed = 0
    # Loop through DataFrame and add speed of each droplet
    for i in range(len(df)):
        total_speed += np.sqrt(df['v_x'][i]**2 + df['v_y'][i]**2 + df['v_z'][i]**2)
    # Divide by number of droplets
    return total_speed/len(df)

def plot_average_droplet_speed(df,dt,ax):
    """
    Plots average speed of droplets over time
    
    Inputs
    -------
    df: pd.DataFrame
        Full dataset
    dt: float
        Length of timestep
    ax: mpl.axes._axes.Axes
        Axes to plot graph on
    """
    # Find start and end times
    end_time = df['time'][len(df)-1]
    time = df['time'][0]
    # Create lists to store speed and time data
    df_temp = extract_timestep(df,time)
    # Only include droplets smaller than original
    df_temp = df_temp[df_temp['vol_t']<2.0].reset_index(drop=True)
    speeds=[]
    times = []
    # Make sure timestep is nonempty else average speed calculation breaks
    if len(df_temp)>0:
        times = [time]
        speeds.append(average_speed(df_temp))
    time += dt
    # Loop through DataFrame and append the times and speeds to relevant list
    while time <= end_time+0.5*dt:
        df_temp = extract_timestep(df,time)
        df_temp = df_temp[df_temp['vol_t']<2.0].reset_index(drop=True)
        if len(df_temp)>0:
            speeds.append(average_speed(df_temp))
            times.append(time)
        time += dt
    # Plot
    ax.plot(times,speeds)
    
def plot_volume_speed(df,ax):
    """
    Produces a scatterplot of log-volumes against speeds for a given dataset
    
    Inputs
    -------
    df: pd.DataFrame
        Full dataset
    ax: mpl.axes._axes.Axes
        Axes to plot graph on
    """
    # Create relevant lists
    speeds = []
    log_vols = []
    # Loop through DataFrame appending speeds and log volumes to relevant list
    for i in range(len(df)):
        speeds.append(np.sqrt(df['v_x'][i]**2 + df['v_y'][i]**2 + df['v_z'][i]**2))
        log_vols.append(np.log10(df['vol_t'][i]))
    # Plot
    ax.scatter(log_vols,speeds,marker='.')
    
def hist_3d(df,fig,ax1,ax2,gs):
    """
    Produces a 3D Density Histogram of the log droplet volumes over time
    Also gives a top view of the histogram. Bars are coloured based on density
    
    AI assisted
    
    Inputs
    -------
    df: pd.DataFrame
        Full dataset
    fig: plt.figure
        Figure
    ax1: mpl_toolkits.mplot3d.axes3d.Axes3D
        Axes to plot 3d graph on
    ax2: mpl.axes._axes.Axes
        Axes to plot top view of density on
    gs: mpl.gridspec.GridSpec
        Colour reference bar
    """
    # Only consider volumes less than original droplet
    df_new = df
    df_new = df[df['vol_t'] < 2.0]
    # Start count later so multiple droplets are available
    # Otherwise density later on is indecipherable
    df_new = df_new[df_new['time']>0.4]
    volumes = np.log10(np.array(df_new['vol_t']))
    times = np.array(df_new['time'])
    unique_times = np.unique(times)
    # Setup for histogram creation
    n_bins = 50
    vol_range = (volumes.min(),volumes.max())
    bin_edges = np.linspace(vol_range[0], vol_range[1], n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
 
    ax2 = plt.gca()
    ax2.set_facecolor('C0')

    # Create Histogram for each slice
    hist_matrix = np.zeros((len(unique_times), n_bins))
    for i, t in enumerate(unique_times):
        vols_at_t = volumes[times == t]
        # density=True normalizes each slice so slices with different
        # droplet counts remain comparable
        hist, _ = np.histogram(vols_at_t, bins=bin_edges, density=True)
        hist_matrix[i] = hist  
    n_bins = len(bin_centers)
    dx = (bin_centers[1] - bin_centers[0]) * 0.9
    # Thickness of each time slice (auto-scaled to time spacing)
    dt = (unique_times[1] - unique_times[0]) * 0.9 if len(unique_times) > 1 else 1.0
    # Create colourmap for histogram bars
    cmap = clr.LinearSegmentedColormap.from_list('gradient',['C0','C1'])
    max_height = hist_matrix.max()
    min_height = hist_matrix.min()
    # Put Histograms together in 3D
    for i, t in enumerate(unique_times):
        xs = bin_centers
        ys = np.full(n_bins, t)
        zs = np.zeros(n_bins)
        dz = hist_matrix[i]
        colour = cmap((dz-min_height)/max_height)
        ax1.bar3d(xs - dx / 2, ys - dt / 2, zs, dx, dt, dz, color=colour)
        ax2.scatter(xs-dx/2,ys-dt/2,marker='s',color=colour,s=100)
    
    # Create colour bar and label axes
    norm = clr.Normalize(vmin=min_height,vmax=max_height)
    sm = cm.ScalarMappable(norm=norm,cmap=cmap)
    sm.set_array([])
    cax = fig.add_subplot(gs[0,2])
    fig.colorbar(sm,cax=cax,cmap=cmap)
    ax1.set_xlabel("Droplet volume")
    ax1.set_ylabel("Time")
    ax1.set_zlabel("Density")
    ax2.set_xlabel("Droplet Volume")
    ax2.set_ylabel("Time")

def freq_3d(df,fig,ax1,ax2,gs):
    """
    Produces a 3D Frequency Histogram of the log droplet volumes over time
    Also gives a top view of the histogram. Bars are coloured based on density
    
    AI assisted
    
    Inputs
    -------
    df: pd.DataFrame
        Full dataset
    fig: plt.figure
        Figure
    ax1: mpl_toolkits.mplot3d.axes3d.Axes3D
        Axes to plot 3d graph on
    ax2: mpl.axes._axes.Axes
        Axes to plot top view of density on
    gs: plt.gridspec.GridSpec
        Colour reference bar
    """
    # Only consider volumes less than original droplet
    df_new = df
    df_new = df[df['vol_t'] < 2.0]
    volumes = np.log10(np.array(df_new['vol_t']))
    times = np.array(df_new['time'])
    unique_times = np.unique(times)
    # Setup for histogram creation
    n_bins = 50
    vol_range = (volumes.min(),volumes.max())
    bin_edges = np.linspace(vol_range[0], vol_range[1], n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax2 = plt.gca()
    ax2.set_facecolor('C0')
 
    # Create Histogram for each slice
    hist_matrix = np.zeros((len(unique_times), n_bins))
    for i, t in enumerate(unique_times):
        vols_at_t = volumes[times == t]
        # density=False creates Frequency graph
        hist, _ = np.histogram(vols_at_t, bins=bin_edges, density=False)
        hist_matrix[i] = hist  
    n_bins = len(bin_centers)
    dx = (bin_centers[1] - bin_centers[0]) * 0.9
    # Thickness of each time slice (auto-scaled to time spacing)
    dt = (unique_times[1] - unique_times[0]) * 0.9 if len(unique_times) > 1 else 1.0
    # Create colourmap for histogram bars
    cmap = clr.LinearSegmentedColormap.from_list('gradient',['C0','C1'])
    max_height = hist_matrix.max()
    min_height = hist_matrix.min()
    # Put Histograms together in 3D
    for i, t in enumerate(unique_times):
        xs = bin_centers
        ys = np.full(n_bins, t)
        zs = np.zeros(n_bins)
        dz = hist_matrix[i]
        colour = cmap((dz-min_height)/max_height)
        ax1.bar3d(xs - dx / 2, ys - dt / 2, zs, dx, dt, dz, color=colour)
        ax2.scatter(xs-dx/2,ys-dt/2,marker='s',color=colour,s=100)
 
    # Create colour bar and label axes
    norm = clr.Normalize(vmin=min_height,vmax=max_height)
    sm = cm.ScalarMappable(norm=norm,cmap=cmap)
    sm.set_array([])
    cax = fig.add_subplot(gs[0,2])
    fig.colorbar(sm,cax=cax,cmap=cmap)
    ax1.set_xlabel("Droplet volume")
    ax1.set_ylabel("Time")
    ax1.set_zlabel("Frequency")
    ax2.set_xlabel("Droplet Volume")
    ax2.set_ylabel("Time")