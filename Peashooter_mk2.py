# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:23:29 2026

@author: wilfl


"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation

# Initial Conditions
R = 1
g = np.array([0,0,-9.81])
dt = 0.01

def initialise(n):
    """
    Sets initial positions of droplets.
    
    Inputs
    -------
    n: int
        Number of droplets on each side
        
    Returns
    -------
    d: pd.DataFrame
        DataFrame containing the droplets' data'
    """
    # Angle and distance between each droplet
    angle = np.pi/(4*(n+1))
    dist = R*angle
    # Max volume of each particle
    max_vol = 4/3*np.pi*(dist/2)**3
    # Set arrays for volumes, velocities, volumes and droplet IDs
    xs, ys, zs = np.zeros(2*n),np.zeros(2*n),np.zeros(2*n)
    v_xs,v_ys,v_zs = np.zeros(2*n),np.zeros(2*n),np.zeros(2*n)
    vol_1s, vol_2s = np.zeros(2*n), np.zeros(2*n)
    droplet_IDs = np.array(range(2*n))
    # Loop through the points and set initial positions, velocities and volumes
    for i in range(n):
        # Random trajectory angle
        traj_angle = np.pi/4 + np.random.normal(0,0.1)
        xs[i] = R*np.cos(3*np.pi/8 + i*angle)
        xs[i+n] = R - xs[i]
        ys[i] = R*np.sin(3*np.pi/8 + i*angle)
        ys[i+n] = ys[i]
        v_xs[i] = 5*R*np.cos(traj_angle)*xs[i]
        v_xs[i+n] = -v_xs[i]
        v_ys[i] = 5*R*np.cos(traj_angle)*ys[i]
        v_ys[i+n] = v_ys[i]
        v_zs[i] = 5*R*np.sin(traj_angle)
        v_zs[i+n]=v_zs[i]
        # Random volumes
        vol_1s[i] = np.random.uniform(0,max_vol)
        vol_2s[i] = np.random.uniform(0,max_vol)
        vol_1s[i+n] = np.random.uniform(0,max_vol)
        vol_2s[i+n] = np.random.uniform(0,max_vol)
    vol_total = vol_1s + vol_2s
    # Put all into DataFrame
    d = {'time': np.zeros(2*n), 'droplet_ID': droplet_IDs, 'x': xs, 'y': ys, 'z': zs,
         'v_x': v_xs, 'v_y': v_ys, 'v_z': v_zs, 'vol_1': vol_1s, 'vol_2': vol_2s, 'vol_t': vol_total}
    d = pd.DataFrame(d)
    return d

def next_pos(pos, vel, dt=0.01):
    """
    Updates a droplet to its next position using projectile motion
    
    Inputs
    -------
    pos: array
        Centres of masses of droplets in Cartesian coordinates
    vel: array
        Velocities of the droplets in Cartesian coordinates
    dt: float
        (optional) Length of timestep
        
    Returns
    -------
    new_pos: array
        New droplet positions
    new_vel: array
        New droplet velocities
    """
    # Find displacement
    displacement = vel*dt + 0.5*(dt**2)*g
    # Update position and velocity
    new_pos = pos + displacement
    new_vel = vel + g*dt
    return new_pos, new_vel

def update(df, dt=0.01):
    """
    Updates the full DataFrame to next timestep
    
    Inputs
    -------
    df: pd.DataFrame
        DataFrame of the droplet data
    dt: float
        (optional) Length of timestep
    """
    n = df.shape[0]
    xs, ys, zs = np.zeros(n),np.zeros(n),np.zeros(n)
    v_xs,v_ys,v_zs = np.zeros(n),np.zeros(n),np.zeros(n)
    # Update position + velocity values in the DataFrame
    for i in range(n):
        pos = np.array([df['x'][i],df['y'][i],df['z'][i]])
        vel = np.array([df['v_x'][i],df['v_y'][i],df['v_z'][i]])
        new_pos, new_vel = next_pos(pos, vel, dt)
        xs[i], ys[i], zs[i] = new_pos
        v_xs[i], v_ys[i], v_zs[i] = new_vel
    # Construct new DataFrame
    d = {'time': (df['time'][0]+dt)*np.ones(n), 'droplet_ID': np.array(df['droplet_ID']), 'x': xs, 'y': ys, 'z': zs,
         'v_x': v_xs, 'v_y': v_ys, 'v_z': v_zs, 'vol_1': df['vol_1'], 'vol_2': df['vol_2'], 'vol_t': df['vol_t']}
    new_df = pd.DataFrame(d)
    del d
    return new_df

def check_zpos(df):
    # Check whether the droplets have reached 
    df_new = df.loc[df['z'] > 0].reset_index(drop=True)
    return df_new

def check_merge(drop1, drop2):
    r1 = (3*drop1['vol_t']/(4*np.pi))**(1/3)
    r2 = (3*drop2['vol_t']/(4*np.pi))**(1/3)
    R = r1 + r2
    dist = (drop1['x']-drop2['x'])**2 + (drop1['y']-drop2['y'])**2 + (drop1['z']-drop2['z'])**2
    dist = np.sqrt(dist)
    
    if dist < R:
        return True
    else:
        return False
    
def merge(df,i,j):
    df.loc[i,'x'] = (df.loc[i,'x']+df.loc[j,'x'])/2
    df.loc[i,'y'] = (df.loc[i,'y']+df.loc[j,'y'])/2
    df.loc[i,'z'] = (df.loc[i,'z']+df.loc[j,'z'])/2
    
    vol_t = df.loc[i,'vol_t'] + df.loc[j,'vol_t']
    df.loc[i,'v_x'] = (df.loc[i,'v_x']*df.loc[i,'vol_t'] + df.loc[j,'v_x']*df.loc[j,'vol_t'])/vol_t
    df.loc[i,'v_y'] = (df.loc[i,'v_y']*df.loc[i,'vol_t'] + df.loc[j,'v_y']*df.loc[j,'vol_t'])/vol_t
    df.loc[i,'v_z'] = (df.loc[i,'v_z']*df.loc[i,'vol_t'] + df.loc[j,'v_z']*df.loc[j,'vol_t'])/vol_t
    
    df.loc[i,'vol_1'] += df.loc[j,'vol_1']
    df.loc[i,'vol_2'] += df.loc[j,'vol_2']
    df.loc[i,'vol_t'] = vol_t