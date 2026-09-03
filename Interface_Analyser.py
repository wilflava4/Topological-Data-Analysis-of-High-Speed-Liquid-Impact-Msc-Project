# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:13:05 2026

@author: wilfl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import ReadFiles as RF



def extract_slice(df,nth_slice,total_n=360):
    """
    Takes a 3D sector from point cloud and projects it onto a 2D slice
    
    Inputs
    -------
    df: pd.DataFrame
        Full point cloud
    nth_slice: int
        Desired sector, 180 degree plane is split into N equal sectors,
        nth_slice counts anticlockwise from 0 starting at slice 90 degrees below horizontal
    total_n: int
        (optional) Total number of sectors, default is 360
    
    Returns
    -------
    df_slice: pd.DataFrame
        DataFrame of the new coordinates on the 2D slice
    tree: scipy.spatial._ckdtree.cKDTree
        KDTree of the coordinates
    coords: np.ndarray
        numpy array of the coordinates
    """
    # Find angle range and midline of the sector
    angle_range = 2*np.pi*np.array([(180/total_n)*nth_slice - 90, (180/total_n)*nth_slice - (90-180/total_n)])/360
    angle_midpoint = (angle_range[0] + angle_range[1])/2
    # Take points within the desired sector
    if nth_slice != total_n-1:
        df_slice = df[(df['angle'] >= angle_range[0]) & (df['angle'] < angle_range[1])]
        df_slice.reset_index(drop=True,inplace=True)
    else:
        df_slice = df[(df['angle'] >= angle_range[0]) & (df['angle'] <= angle_range[1])]
        df_slice.reset_index(drop=True,inplace=True)
    # Project points to line, details can be found in Methodology section of the report
    for i in range(len(df_slice)):
        r = np.sqrt(df_slice['x'][i]**2 + df_slice['z'][i]**2)
        df_slice.loc[i,'x'] = r*np.cos(angle_midpoint)
        df_slice.loc[i,'z'] = r*np.sin(angle_midpoint)
    
    # Build KDTree
    tree, coords = build_kdtree(df_slice)
    # Remove duplicates
    df_slice = remove_duplicates(df_slice, tree, coords)
    # Rebuild KDTree with reduced dataset
    tree, coords = build_kdtree(df_slice)
    return df_slice, tree, coords

def remove_duplicates(df, tree, coords):
    """
    Removes points in KDTree within tolerance of 1e-3 of each other
    
    AI assisted
    
    Inputs
    -------
    df: pd.DataFrame
        Point cloud in DataFrame
    tree: scipy.spatial._ckdtree.cKDTree
        KDTree
    coords: np.ndarray
        numpy array of the coordinates
    
    Returns
    -------
    df_new: pd.DataFrame
        DataFrame of reduced slice
    """
    n = len(df)
    pairs = tree.query_pairs(r=1e-3, p=np.inf)

    # Union-Find to group chains of near-duplicates (A~B~C even if A/C aren't directly within tol)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, j in pairs:
        union(i, j)

    # Keep only the first occurrence (lowest index) within each group
    roots = np.array([find(i) for i in range(n)])
    _, first_idx = np.unique(roots, return_index=True)

    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[first_idx] = True

    return df[keep_mask].reset_index(drop=True)

def plot_slice(df_slice,ax):
    """
    Plots a given slice
    
    Inputs
    -------
    df_slice: pd.DataFrame
        Point cloud of slice to be plotted
    ax: mpl.axes._axes.Axes
        Axes to plot point cloud on
    """
    # Find distances of points from y-axis
    x_coords = np.sqrt(np.array(df_slice['x'])**2+np.array(df_slice['z'])**2)
    # Plot
    ax.scatter(x_coords,df_slice['y'],marker='.')

def build_kdtree(df):
    """
    Builds a KDTree from given DataFrame
    
    Inputs
    -------
    df: pd.DataFrame
        Point cloud to be converted to KDTree
    
    Returns
    -------
    tree: scipy.spatial._ckdtree.cKDTree
        KDTree of points
    coords: np.ndarray
        numpy array of the coordinates
    """
    coords = df[['x', 'y', 'z']].to_numpy()
    return cKDTree(coords), coords

def find_roots(df,tree,coords,start_point1,start_point2=None,
               prev_known=False):
    """
    Finds the roots of the jet for a droplet landing on a pool
    
    Inputs
    -------
    df: pd.DataFrame
        DataFrame of point cloud
    tree: scipy.spatial._ckdtree.cKDTree
        KDTree of points
    coords: np.ndarray
        numpy array of point cloud coordinates
    start_point1: np.ndarray, shape = (1,3)
        Start point for search of the left root. Should be placed approximately
        near droplet edge above the root
    start_point2: np.ndarray, shape = (1,3)
        (optional) start point for search of the right root,
        must be given if and only if prev_known==True
    prev_known: bool
        (optional) indicates whether the start points are from roots at previous timestep
    
    Returns
    -------
    root1_point: np.ndarray, shape = (1,3)
        Coordinate of the left root
    root2_point: np.ndarray, shape = (1,3)
        Coordinate of the right root
    """
    # Algorithm if previous roots are known
    if prev_known:
        # Find 4 nearest points to previous left root
        n_indices = tree.query(start_point1,k=4)[1]
        min_height = 100
        # Find point with lowest y-coordinate
        for i in n_indices:
            prev_indices.append(i)
            height = coords[i][1]
            if height < min_height:
                index = i
                min_height = height
        # Return lowest point
        root1_point = coords[index]
        
        # Find 4 nearest points to previous right root
        n_indices = tree.query(start_point2,k=4)[1]
        points = coords[n_indices]
        # Sort from lowest to highest
        points = points[points[:,1].argsort()]
        # Find gradient between 2 lowest points
        prev_point = points[0]
        point = points[1]
        i = 1
        numerator = point[1] - prev_point[1]
        denominator = np.sqrt(point[0]**2 + point[2]**2)-np.sqrt(prev_point[0]**2 + prev_point[2]**2)
        # Avoid 0 division
        if abs(denominator < 1e-10):
            gradient = -10
        else:
            gradient = numerator/denominator
        # Find gradients between points until the gradient is greater than -1
        while gradient > -1:
            if i<3:
                prev_point = points[i]
                point = points[i+1]
                i += 1
            else:
                prev_point = point
            numerator = point[1] - prev_point[1]
            denominator = np.sqrt(point[0]**2 + point[2]**2)-np.sqrt(prev_point[0]**2 + prev_point[2]**2)
            if abs(denominator < 1e-10):
                gradient = -10
            else:
                gradient = numerator/denominator
        # Return root at which gradient goes greater than -1 or final of 4 found points
        root2_point = prev_point
        
    # Algorithm for unknown previous roots
    else:
        # Find 2 points closest to the start point
        n_indices = tree.query(start_point1,k=2)[1]
        point = coords[n_indices[0]]
        # Begin list of visited points
        prev_indices = [n_indices[0]]
        neighbour = coords[n_indices[1]]
        max_height = point[1]
        n = 3
        # Make sure search traverses downwards
        while neighbour[1] > point[1]:
            n_indices = tree.query(point,k=n)[1]
            prev_indices.append(n_indices[-1])
            neighbour = coords[n_indices[-1]]
            n += 1
        # Search until next neighbour is above the previous
        while neighbour[1] <= max_height:
            point = neighbour.copy()
            max_height = point[1]
            n_indices = tree.query(point,k=2)[1]
            
            # Find next coordinate not already searched
            n = 3
            while set(n_indices).issubset(prev_indices):
                n_indices = tree.query(point,k=n)[1]
                n += 1
            # Append this index to list of searched indices
            for i in n_indices:
                if i not in prev_indices:
                    neighbour = coords[i]
                    prev_indices.append(i)
                    break
        # Left root is lowest searched point
        root1_point = point.copy()

        # Begin search for right root 5% further out and 3% further lower
        # This start point is only robust for early jet formation
        # To begin later, begin search from further out
        starty = 0.97*root1_point[1]
        start_point2 = 1.05*root1_point
        start_point2[1] = starty
        n_indices = tree.query(start_point2,k=2)[1]
        point = coords[n_indices[0]]
        neighbour = coords[n_indices[1]]
        prev_indices.append(n_indices[0])
        prev_indices.append(n_indices[1])
        # Find distance from y-axis
        x_star = point[0]**2 + point[2]**2
        next_x = neighbour[0]**2 + neighbour[2]**2
        n = 3
        # Begin by searching towards the y-axis
        while next_x > x_star:
            n_indices = tree.query(point,k=n)[1]
            prev_indices.append(n_indices[-1])
            neighbour = coords[n_indices[-1]]
            next_x = neighbour[0]**2 + neighbour[2]**2
            n += 1
        numerator = point[1] - neighbour[1]
        denominator = x_star - next_x
        # Find gradient between neighbouring point
        if abs(denominator) < 1e-10:
            gradient = -10
        else:
            gradient = numerator/denominator
        # Continue until gradient > -1
        while gradient > -1:
            point = neighbour.copy()
            n_indices = tree.query(point,k=2)[1]
            neighbour = coords[n_indices[-1]]
            n = 3
            while set(n_indices).issubset(prev_indices):
                n_indices = tree.query(point,k=n)[1]
                n += 1
            for i in n_indices:
                if i not in prev_indices:
                    neighbour = coords[i]
                    prev_indices.append(i)
                    break
            numerator = point[1] - neighbour[1]
            denominator = np.sqrt(point[0]**2 + point[2]**2)-np.sqrt(neighbour[0]**2 + neighbour[2]**2)
            if abs(denominator) < 1e-10:
                gradient = -10
            else:
                gradient = numerator/denominator
        # Right root is last point before gradient goes above -1
        root2_point = point.copy() 
    return root1_point, root2_point
    
def find_roots2(df,tree,coords,start_point1,start_point2=None,
               prev_known=False):
    """
    Finds the roots of the jet for two droplets colliding
    
    Inputs
    -------
    df: pd.DataFrame
        DataFrame of point cloud
    tree: scipy.spatial._ckdtree.cKDTree
        KDTree of points
    coords: np.ndarray
        numpy array of point cloud coordinates
    start_point1: np.ndarray, shape = (1,3)
        Start point for search of the top root. Should be placed approximately
        near droplet edge above the root
    start_point2: np.ndarray, shape = (1,3)
        (optional) start point for search of the bottom root,
        must be given if and only if prev_known==True
    prev_known: bool
        (optional) indicates whether the start points are from roots at previous timestep
    
    Returns
    -------
    root1_point: np.ndarray, shape = (1,3)
        Coordinate of the top root
    root2_point: np.ndarray, shape = (1,3)
        Coordinate of the bottom root
    """
    # Algorithm if previous roots are known
    if prev_known:
        # Find 4 nearest points to previous top root
        n_indices = tree.query(start_point1,k=4)[1]
        min_x = 100
        # Find point closest to y-axis
        for i in n_indices:
            x_star = coords[i][0]**2 + coords[i][2]**2
            if x_star < min_x:
                index = i
                min_x = x_star
        # Top root is closest point to y-axis
        root1_point = coords[index]
        
        # Find 4 points closest to bottom root
        n_indices = tree.query(start_point2,k=4)[1]
        min_x = 100
        # Find point closest to y-axis
        for i in n_indices:
            x_star = coords[i][0]**2 + coords[i][2]**2
            if x_star < min_x:
                index = i
                min_x = x_star
        # Bottom root is closest point to y-axis
        root2_point = coords[index]
    
    # Algorithm if previous roots are not known
    else:
        # Find 2 closest points to start point
        n_indices = tree.query(start_point1,k=2)[1]
        index = n_indices[0]
        point = coords[index]
        prev_indices = [index]
        prev_indices.append(n_indices[1])
        neighbour = coords[n_indices[1]]
        n = 3
        # Ensure search traverses towards the y-axis
        while neighbour[1] > point[1]:
            n_indices = tree.query(point,k=n)[1]
            prev_indices.append(n_indices[-1])
            neighbour = coords[n_indices[-1]]
            n += 1
        min_x = point[0]**2 + point[2]**2
        x_star = neighbour[0]**2 + neighbour[2]**2
        b = 0
        # Find last point before next is further away from the y-axis
        while x_star <= min_x:
            b += 1
            min_x = x_star
            prev_index = index
            point = neighbour.copy()
            n_indices = tree.query(point,k=n)[1]
            neighbour = coords[n_indices[1]]
            n = 3
            # Make sure next point has not yet been searched
            while set(n_indices).issubset(prev_indices) or neighbour[1] > point[1]:
                n_indices = tree.query(point,k=n)[1]
                neighbour = coords[n_indices[-1]]
                n += 1
            for i in n_indices:
                if i not in prev_indices and coords[i][1] < point[1]:
                    index = i
                    neighbour = coords[i]
                    prev_indices.append(i)
                    break
            x_star = neighbour[0]**2 + neighbour[2]**2
        # Top root is closest searched point to y-axis
        root1_index = prev_index
        root1_point = coords[root1_index]
        
        # Begin search for bottom root at same x-z coord as top root
        # but y-coord at 5-y
        start_point2 = root1_point.copy()
        start_point2[1] = 5 - root1_point[1]
        n_indices = tree.query(start_point2,k=4)[1]
        min_x = 100
        # Find closest point to y-axis in same manner
        for i in n_indices:
            temp = coords[i]
            x_star = temp[0]**2 + temp[2]**2
            if x_star < min_x and i not in prev_indices:
                index = i
                min_x = x_star
        root2_point = coords[index]
    return root1_point, root2_point

def plot_features(df,nth_slice,start_point1=[0,5,0],start_point2=[0,0,0],prev_known=False):
    """
    Plots the features of the roots of the jet and the centre line for
    a droplet on a pool for a given timestep and specified slice
    
    Inputs
    -------
    df: pd.DataFrame
        Full point cloud
    nth_slice: int
        Specified slice to find and plot features from
    start_point1: np.ndarray, shape = (1,3)
        Start point for search of the top root. Should be placed approximately
        near droplet edge above the root
    start_point2: np.ndarray, shape = (1,3)
        (optional) start point for search of the bottom root,
        must be given if and only if prev_known==True
    prev_known: bool
        (optional) indicates whether the start points are from roots at previous timestep
    """
    # Extract slice
    df_slice, tree, coords = extract_slice(df,nth_slice)
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot()
    # Plot the slice
    plot_slice(df_slice,ax)
    # Find the left and right roots then plot them
    left, right= find_roots(df_slice,tree,coords,start_point1,start_point2,prev_known)
    ax.scatter([np.sqrt(left[0]**2+left[2]**2),np.sqrt(right[0]**2+right[2]**2)],[left[1],right[1]])
    # Find the centre line and plot it
    midline = centre_line(df_slice,tree,coords,left,right)[0]
    ax.plot(np.sqrt(midline[:,0]**2+midline[:,2]**2),midline[:,1],color='C1')
    plt.show()
    
def find_all_roots(df, prev_known=False,prev_left=None,prev_right=None):
    """
    Finds all roots for a given timestep for a droplet on a pool. Specifically
    uses 360 total slices. Code interpolates roots to fit approximately on
    an ellipse, the number of interpolated points is printed to the console
    
    Inputs
    -------
    df: pd.DataFrame
        Full point cloud
    prev_known: bool
        (optional) signals whether the roots from a previous timestep
        are to be used
    prev_left: np.ndarray, shape = (360,3)
        (optional) left roots from previous timestep
    prev_right: np.ndarray, shape = (360,3)
        (optional) right roots from previous timestep
    
    Returns
    -------
    left_roots: np.ndarray, shape = (360,3)
        left roots
    right_roots: np.ndarray, shape = (360,3)
        right roots
    """
    # Create lists to store roots in
    left_roots = []
    right_roots = []
    # Extract each slice and find the left and right roots
    # appending them to relevant lists
    for i in range(360):
        df_slice, tree, coords = extract_slice(df,i)
        if prev_known:
            left, right = find_roots(df_slice,tree,coords,prev_left[i],prev_right[i],prev_known)
        else:
            left,right = find_roots(df_slice,tree,coords,[0,5,0])
        left_roots.append(left)
        right_roots.append(right)
    # Convert lists to numpy arrays
    left_roots = np.array(left_roots)
    right_roots = np.array(right_roots)
    
    # Fit ellipse to left and right roots in x-z plane
    al,bl,cl,dl,el,fl = fit_ellipse(left_roots[:,0],left_roots[:,2])
    ar,br,cr,dr,er,fr = fit_ellipse(right_roots[:,0],right_roots[:,2])
    # Find median heights of roots
    left_y = np.median(left_roots[:,1])
    right_y = np.median(right_roots[:,1])
    left_count = 0
    right_count = 0
    # Loop through points and check whether they lie outside ellipse tolerance
    # if so, place on ellipse (details included in report) and count number
    # of interpolation occurrences
    for j in range(360):
        xl = left_roots[j][0]
        zl = left_roots[j][2]
        if abs(al*xl**2+bl*xl*zl+cl*zl**2+dl*xl+el*zl+fl) > 2e-3:
            z_coeff = np.tan(2*np.pi*(0.5*j-89.75)/360)
            left_roots[j][0] = np.max(np.roots([al+bl*z_coeff+cl*z_coeff**2,dl+el*z_coeff,fl]))
            left_roots[j][2] = z_coeff * left_roots[j][0]
            left_roots[j][1] = left_y
            left_count += 1
        xr = right_roots[j][0]
        zr = right_roots[j][2]
        if abs(ar*xr**2+br*xr*zr+cr*zr**2+dr*xr+er*zr+fr) > 2e-3:
            z_coeff = np.tan(2*np.pi*(0.5*j-89.75)/360)
            right_roots[j][0] = np.max(np.roots([ar+br*z_coeff+cr*z_coeff**2,dr+er*z_coeff,fr]))
            right_roots[j][2] = z_coeff * right_roots[j][0]
            right_roots[j][1] = right_y
            right_count += 1
    
    # Print number of times roots were interpolated
    print(f'Left Roots Interpolated: {left_count}')
    print(f'Right Roots Interpolated: {right_count}')
    return left_roots, right_roots
                
def find_all_roots2(df,prev_known=False,prev_top=None,prev_bottom=None):
    """
    Finds all roots for a given timestep for two colliding droplets. Specifically
    uses 360 total slices. Code interpolates roots to fit approximately on
    an ellipse, the number of interpolated points is printed to the console
    
    Inputs
    -------
    df: pd.DataFrame
        Full point cloud
    prev_known: bool
        (optional) signals whether the roots from a previous timestep
        are to be used
    prev_left: np.ndarray, shape = (360,3)
        (optional) top roots from previous timestep
    prev_right: np.ndarray, shape = (360,3)
        (optional) bottom roots from previous timestep
    
    Returns
    -------
    top_roots: np.ndarray, shape = (360,3)
        Top roots
    bottom_roots: np.ndarray, shape = (360,3)
        Bottom roots
    """
    # Recycled code from droplet on pool example so left = top and right = bottom
    # throughout the function, only final names changed
    
    # Create lists to store roots in
    left_roots = []
    right_roots = []
    prev_left = prev_top
    prev_right = prev_bottom
    # Extract each slice and find the left and right roots
    # appending them to relevant lists
    for i in range(360):
        df_slice,tree,coords = extract_slice(df,i)
        if prev_known:
            left, right = find_roots2(df_slice,tree,coords,prev_left[i],prev_right[i],prev_known)
        else:
            x_coord = 0.8*np.cos(2*np.pi*(0.5*i-89.75)/360)
            z_coord = 0.8*np.sin(2*np.pi*(0.5*i-89.75)/360)
            left,right = find_roots2(df_slice,tree,coords,[x_coord,3.0,z_coord])
        left_roots.append(left)
        right_roots.append(right)
        
    # Convert lists to numpy arrays
    left_roots = np.array(left_roots)
    right_roots = np.array(right_roots)
    # Fit ellipse to top and bottom roots in x-z plane
    al,bl,cl,dl,el,fl = fit_ellipse(left_roots[:,0],left_roots[:,2])
    ar,br,cr,dr,er,fr = fit_ellipse(right_roots[:,0],right_roots[:,2])
    # Find median heights of roots
    left_y = np.median(left_roots[:,1])
    right_y = np.median(right_roots[:,1])
    left_count = 0
    right_count = 0
    # Loop through points and check whether they lie outside ellipse tolerance
    # if so, place on ellipse (details included in report) and count number
    # of interpolation occurrences
    for j in range(360):
        xl = left_roots[j][0]
        zl = left_roots[j][2]
        if abs(al*xl**2+bl*xl*zl+cl*zl**2+dl*xl+el*zl+fl) > 2e-3:
            z_coeff = np.tan(2*np.pi*(0.5*j-89.75)/360)
            left_roots[j][0] = np.max(np.roots([al+bl*z_coeff+cl*z_coeff**2,dl+el*z_coeff,fl]))
            left_roots[j][2] = z_coeff * left_roots[j][0]
            left_roots[j][1] = left_y
            left_count += 1
        xr = right_roots[j][0]
        zr = right_roots[j][2]
        if abs(ar*xr**2+br*xr*zr+cr*zr**2+dr*xr+er*zr+fr) > 2e-3:
            z_coeff = np.tan(2*np.pi*(0.5*j-89.75)/360)
            right_roots[j][0] = np.max(np.roots([ar+br*z_coeff+cr*z_coeff**2,dr+er*z_coeff,fr]))
            right_roots[j][2] = z_coeff * right_roots[j][0]
            right_roots[j][1] = right_y
            right_count += 1
            
    # Print number of times roots were interpolated
    print(f'Top Roots Interpolated: {left_count}')
    print(f'Bottom Roots Interpolated: {right_count}')
    top_roots = left_roots
    bottom_roots = right_roots
    return top_roots, bottom_roots

def do_it_all():
    """
    Finds all roots for all frames for a droplet on a pool. Code uses data
    from a specific dataset which is read in from a different python file.
    The frames considered are hard coded, this would have to be rewritten
    for a new dataset
    """
    
    # Find all roots for all timesteps and write them all to a file
    df = RF.get_DropPool3(37)
    left, right = find_all_roots(df)
    left = np.hstack((37*np.ones(len(left)).reshape(len(left),1),left))
    right = np.hstack((37*np.ones(len(right)).reshape(len(right),1),right))
    with open('D:\DropPool3\\left_roots.txt','a') as f:
        np.savetxt(f, left,fmt = ('% 4d', '%1.5f','%1.5f','%1.5f'),delimiter = ' ')
        f.write('\n')
    with open('D:\DropPool3\\right_roots.txt','a') as f:
        np.savetxt(f, right,fmt = ('% 4d', '%1.5f','%1.5f','%1.5f'),delimiter = ' ')
        f.write('\n')
    # Print when each step is done
    print(37)
    for j in range(38,73):
        df = RF.get_DropPool3(j)
        left, right = find_all_roots(df,True,left[:,1:4],right[:,1:4])
        left = np.hstack((j*np.ones(len(left)).reshape(len(left),1),left))
        right = np.hstack((j*np.ones(len(right)).reshape(len(right),1),right))
        with open('D:\DropPool3\\left_roots.txt','a') as f:
            np.savetxt(f, left,fmt = ('% 4d', '%1.5f','%1.5f','%1.5f'),delimiter = ' ')
            f.write('\n')
        with open('D:\DropPool3\\right_roots.txt','a') as f:
            np.savetxt(f, right,fmt = ('% 4d', '%1.5f','%1.5f','%1.5f'),delimiter = ' ')
            f.write('\n')
        print(j)
        
def do_it_all2():
    """
    Finds all roots for all frames for two colliding droplets. Code uses data
    from a specific dataset which is read in from a different python file.
    The frames considered are hard coded, this would have to be rewritten
    for a new dataset
    """
    
    # Find all roots for all timesteps and write them all to a file
    df = RF.get_2Drops2(27)
    left, right = find_all_roots2(df)
    left = np.hstack((27*np.ones(len(left)).reshape(len(left),1),left))
    right = np.hstack((27*np.ones(len(right)).reshape(len(right),1),right))
    with open('D:\\TwoDrops2\\left_roots2.txt','a') as f:
        np.savetxt(f, left,fmt = ('% 4d', '%1.5f','%1.5f','%1.5f'),delimiter = ' ')
        f.write('\n')
    with open('D:\\TwoDrops2\\right_roots2.txt','a') as f:
        np.savetxt(f, right,fmt = ('% 4d', '%1.5f','%1.5f','%1.5f'),delimiter = ' ')
        f.write('\n')
    # Print when each step is done
    print(27)
    for j in range(28,65):
        df = RF.get_2Drops2(j)
        left, right = find_all_roots2(df,True,left[:,1:4],right[:,1:4])
        left = np.hstack((j*np.ones(len(left)).reshape(len(left),1),left))
        right = np.hstack((j*np.ones(len(right)).reshape(len(right),1),right))
        with open('D:\\TwoDrops2\\left_roots2.txt','a') as f:
            np.savetxt(f, left,fmt = ('% 4d', '%1.5f','%1.5f','%1.5f'),delimiter = ' ')
            f.write('\n')
        with open('D:\\TwoDrops2\\right_roots2.txt','a') as f:
            np.savetxt(f, right,fmt = ('% 4d', '%1.5f','%1.5f','%1.5f'),delimiter = ' ')
            f.write('\n')
        print(j)

def graph_mean_root_distance(df1,df2,start_frame,end_frame,ax):
    """
    Graphs the mean distance between the roots of the jet for a droplet on
    a pool with regression line of a+bt^{3/2}
    
    Inputs
    -------
    df1: pd.DataFrame
        DataFrame containing the coordinates of the left roots for all timesteps
    df2: pd.DataFrame
        DataFrame containing the coordinates of the right roots for all timesteps
    start_frame: int
        First timestep
    end_frame: int
        Final timestep
    ax: mpl.axes._axes.Axes
        Axes to plot graph on
    """
    
    # Set up list to contain the mean distances
    mean_dists = []
    # Extract the root data for each specific time frame and find mean distance
    # between the roots
    for i in range(start_frame,end_frame+1):
        data1 = df1[df1['time']==i]
        data2 = df2[df2['time']==i]
        dist = np.sqrt((data1['x']-data2['x'])**2 + (data1['y']-data2['y'])**2+(data1['z']-data2['z'])**2)
        mean_dists.append(np.mean(dist))
        
    # Plot the mean root distances
    ax.plot(np.array(range(start_frame,end_frame+1))/100,mean_dists,'o-',label='Mean Root Distance')
    mean_dists = np.array(mean_dists)
    
    # Create and plot regression lines
    times = (np.array(range(start_frame,end_frame+1))/100)**(3/2)
    A = np.hstack((np.ones(len(mean_dists)).reshape(len(mean_dists),1),times.reshape(len(mean_dists),1)))
    x = np.linalg.lstsq(A,mean_dists)[0]
    xs = np.linspace(start_frame/100,end_frame/100)
    regression = lambda y: x[0] + x[1]*y**(3/2)
    x[0] = '%.3f' % x[0]
    x[1] = '%.3f' % x[1]
    ax.plot(xs,regression(xs), label = f'Regression, y={x[0]}$+${x[1]}$\sqrt{{t}}^3$',linestyle='dashed')
    print(np.corrcoef(mean_dists,times,rowvar=True))
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance Between Jet Roots')
    ax.legend()
    
def graph_mean_root_distance2(df1,df2,start_frame,end_frame,ax):
    """
    Graphs the mean distance between the roots of the jet for two droplets
    colliding with regression line of a+bt^3
    
    Inputs
    -------
    df1: pd.DataFrame
        DataFrame containing the coordinates of the top roots for all timesteps
    df2: pd.DataFrame
        DataFrame containing the coordinates of the bottom roots for all timesteps
    start_frame: int
        First timestep
    end_frame: int
        Final timestep
    ax: mpl.axes._axes.Axes
        Axes to plot graph on
    """
    
    # Set up list to contain the mean distances   
    mean_dists = []
    # Extract the root data for each specific time frame and find mean distance
    # between the roots
    for i in range(start_frame,end_frame+1):
        data1 = df1[df1['time']==i]
        data2 = df2[df2['time']==i]
        dist = np.sqrt((data1['x']-data2['x'])**2 + (data1['y']-data2['y'])**2+(data1['z']-data2['z'])**2)
        mean_dists.append(np.mean(dist))
        
    # Plot the mean root distances
    ax.plot(np.array(range(start_frame,end_frame+1))/100,mean_dists,'o-',label='Mean Root Distance')
    mean_dists = np.array(mean_dists)
    
    # Create and plot regression lines
    times = (np.array(range(start_frame,end_frame+1))/100)**3
    A = np.hstack((np.ones(len(mean_dists)).reshape(len(mean_dists),1),times.reshape(len(mean_dists),1)))
    x = np.linalg.lstsq(A,mean_dists)[0]
    xs = np.linspace(start_frame/100,end_frame/100)
    regression = lambda y: x[0] + x[1]*y**3
    x[0] = '%.3f' % x[0]
    x[1] = '%.3f' % x[1]
    ax.plot(xs,regression(xs), label = f'Regression, y={x[0]}$+${x[1]}${{t}}^3$',linestyle='dashed')
    print(np.corrcoef(mean_dists,times,rowvar=True))
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance Between Jet Roots')
    ax.legend()


def centre_line(df, tree, coords, root1, root2):
    """
    Finds centre line of the jet for a droplet on a pool
    
    Inputs
    -------
    df: pd.DataFrame
        DataFrame of point cloud
    tree: scipy.spatial._ckdtree.cKDTree
        KDTree of points
    coords: np.ndarray
        numpy array of point cloud coordinates
    root1: np.ndarray, shape = (1,3)
        Location of left root
    root2: np.ndarray, shape = (1,3)
        Location of right root
    
    Returns
    -------
    midpoints: np.ndarray
        Coordinates of points in centre line
    left_jet: np.ndarray
        Coordinates of points along left side of the jet
    right_jet: np.ndarray
        Coordinates of points along right side of the jet
    """
    # Find indices of the roots
    index1 = tree.query(root1,k=1)[1]
    root1 = coords[index1]
    index2 = tree.query(root2,k=1)[1]
    root2 = coords[index2]
    midpoints = []
    # Find distance from y-axis of left root
    root1_x = np.sqrt(root1[0]**2+root1[2]**2)
    # Find y-coord of right root
    root2_y = root2[1]
    
    # If roots are same point do nothing to avoid crashing
    if np.linalg.norm(root1 - root2) < 1e-6:
        continue_mapping = False
    else:
        continue_mapping = True
    
    # Find midpoint of roots
    point1 = coords[index1]
    point2 = coords[index2]
    midpoints.append((point1 + point2)/2)
    left_indices = [index1]
    right_indices = [index2]

    while continue_mapping:
        # Search to the right of the left root
        n_indices = list(tree.query(point1,k=2)[1])
        n=3
        next1 = coords[n_indices[-1]]
        next1_x = np.sqrt(next1[0]**2+next1[2]**2)
        # Make sure new point is searched and this new point is not from the other jet
        # also make sure new point is further to the right than the left root
        while set(n_indices).issubset(left_indices) and len(set(n_indices+right_indices)) == len(n_indices)+len(right_indices) or (next1_x <= root1_x and n_indices[-1] not in left_indices):
            n_indices = list(tree.query(point1,k=n)[1])
            n += 1
            next1 = coords[n_indices[-1]]
            next1_x = np.sqrt(next1[0]**2+next1[2]**2)
        # Add new point to list of searched left indices
        for i in n_indices:
            temp = coords[i]
            tempx = np.sqrt(temp[0]**2 + temp[2]**2)
            if i not in left_indices:
                left_indices.append(i)
                if tempx > root1_x:
                    index1 = i
                    point1 = coords[i]
                    break
        # Stop search if a found point belongs to the other side of the jet
        if len(set(n_indices+right_indices)) != len(n_indices)+len(right_indices):
            continue_mapping = False
        
        # Search upwards of the right root
        n_indices = list(tree.query(point2,k=2)[1])
        n=3
        next2 = coords[n_indices[-1]]
        next2_y = next2[1]
        # Make sure new point is searched and this new point is not from the other jet
        # also make sure new point is above the right root
        while len(set(left_indices+n_indices)) == len(left_indices)+len(n_indices) and set(n_indices).issubset(right_indices) or next2_y <= root2_y:
            n_indices = list(tree.query(point2,k=n)[1])
            n += 1
            next2 = coords[n_indices[-1]]
            next2_y = next2[1]
        # Add new point to list of searched left indices
        for i in n_indices:
            temp = coords[i]
            if i not in right_indices and temp[1] >= root2_y:
                index2 = i
                point2 = coords[i]
                right_indices.append(i)
                break
        # Stop search if a found point belongs to the other side of the jet
        if len(set(left_indices+n_indices)) != len(left_indices)+len(n_indices):
            continue_mapping=False
        
        midpoints.append((point1 + point2)/2)
            
    return np.array(midpoints), coords[left_indices], coords[right_indices]

def centre_line2(df, tree, coords, root1, root2):
    """
    Finds centre line of the jet for two droplets colliding
    
    Inputs
    -------
    df: pd.DataFrame
        DataFrame of point cloud
    tree: scipy.spatial._ckdtree.cKDTree
        KDTree of points
    coords: np.ndarray
        numpy array of point cloud coordinates
    root1: np.ndarray, shape = (1,3)
        Location of top root
    root2: np.ndarray, shape = (1,3)
        Location of bottom root
    
    Returns
    -------
    midpoints: np.ndarray
        Coordinates of points in centre line
    top_jet: np.ndarray
        Coordinates of points along top side of the jet
    bottom_jet: np.ndarray
        Coordinates of points along bottom side of the jet
    """
    
    # Find indices of the roots
    index1 = tree.query(root1,k=1)[1]
    root1 = coords[index1]
    index2 = tree.query(root2,k=1)[1]
    root2 = coords[index2]
    midpoints = []
    # Find y-coords of roots
    root1_y = root1[1]
    root2_y = root2[1]
    
    # If roots are same point do nothing to avoid crashing
    if np.linalg.norm(root1 - root2) < 1e-6:
        continue_mapping = False
    else:
        continue_mapping = True
    # Find midpoint of the roots
    point1 = coords[index1]
    point2 = coords[index2]
    midpoints.append((point1 + point2)/2)
    left_indices = [index1]
    right_indices = [index2]
    while continue_mapping:
        # Search below the top root
        n_indices = list(tree.query(point1,k=2)[1])
        n=3
        next1 = coords[n_indices[-1]]
        next1_y = next1[1]
        # Make sure new point is searched and this new point is not from the other jet
        # also make sure new point is below the top root
        while set(n_indices).issubset(left_indices) and len(set(n_indices+right_indices)) == len(n_indices)+len(right_indices) or next1_y >= root1_y:
            n_indices = list(tree.query(point1,k=n)[1])
            n += 1
            next1 = coords[n_indices[-1]]
            next1_y = next1[1]
        # Add new point to list of searched top indices
        for i in n_indices:
            temp = coords[i]
            tempy = temp[1]
            if i not in left_indices and tempy <= root1_y:
                index1 = i
                point1 = coords[i]
                left_indices.append(i)
                break
            
        # Stop search if a found point belongs to the other side of the jet
        if len(set(n_indices+right_indices)) != len(n_indices)+len(right_indices):
            continue_mapping = False
            
        # Search above the bottom root
        n_indices = list(tree.query(point2,k=2)[1])
        n=3
        next2 = coords[n_indices[-1]]
        next2_y = next2[1]
        # Make sure new point is searched and this new point is not from the other jet
        # also make sure new point is above the bottom root
        while len(set(left_indices+n_indices)) == len(left_indices)+len(n_indices) and set(n_indices).issubset(right_indices) or next2_y <= root2_y:
            n_indices = list(tree.query(point2,k=n)[1])
            n += 1
            next2 = coords[n_indices[-1]]
            next2_y = next2[1]
        for i in n_indices:
            temp = coords[i]
            if i not in right_indices and temp[1] >= root2_y:
                index2 = i
                point2 = coords[i]
                right_indices.append(i)
                break
        
        # Stop search if a found point belongs to the other side of the jet
        if len(set(left_indices+n_indices)) != len(left_indices)+len(n_indices):
            continue_mapping=False
        
        midpoints.append((point1 + point2)/2)
    midpoints = np.array(midpoints)
            
    return np.array(midpoints), coords[left_indices], coords[right_indices]

def centre_line_evolution(nth_slice):
    """
    Finds all coordinates for the centre lines of jets through the same slice
    over time for a droplet on a pool. Frames are hard coded to a specific dataset.
    
    Inputs
    -------
    nth_slice: int
        Desired slice
    
    Returns
    -------
    Centre_lines: list
        List of arrays of centre line coordinates
    left_jets: list
        List of arrays of left jet coordinates
    right_jets: list
        List of arrays of right jet coordinates
    
    """
    
    # Set up lists to store centre lines and roots
    centre_lines = []
    left_jets = []
    right_jets = []
    # Get root data
    left_df = RF.get_left_roots()
    right_df = RF.get_right_roots()
    
    # For each timeframe, find roots and centre line
    for i in range(37,73):
        df = RF.get_DropPool3(i)
        df_slice, tree, coords = extract_slice(df,nth_slice)
        left = left_df[left_df['time'] == i]
        right = right_df[right_df['time'] == i]
        left = np.array(left)[:,1:4][nth_slice]
        right = np.array(right)[:,1:4][nth_slice]
        a,b,c = centre_line(df_slice,tree,coords,left,right)
        centre_lines.append(a)
        left_jets.append(b)
        right_jets.append(c)
        print(i)
    return centre_lines, left_jets, right_jets

def centre_line_evolution2(nth_slice):
    """
    Finds all coordinates for the centre lines of jets through the same slice
    over time for two droplets colliding. Frames are hard coded to a specific dataset.
    
    Inputs
    -------
    nth_slice: int
        Desired slice
    
    Returns
    -------
   centre_lines: list
        List of arrays of centre line coordinates
    top_jets: list
        List of arrays of top jet coordinates
    bottom_jets: list
        List of arrays of bottom jet coordinates
    
    """
    
    # Set up lists to store centre lines and roots
    centre_lines = []
    left_jets = []
    right_jets = []
    left_df = RF.get_left_roots2()
    right_df = RF.get_right_roots2()
    
    # For each timeframe, find roots and centre line
    for i in range(27,65):
        df = RF.get_2Drops2(i)
        df_slice, tree, coords = extract_slice(df,nth_slice)
        left = left_df[left_df['time'] == i]
        right = right_df[right_df['time'] == i]
        left = np.array(left)[:,1:4][nth_slice]
        right = np.array(right)[:,1:4][nth_slice]
        a,b,c = centre_line2(df_slice,tree,coords,left,right)
        centre_lines.append(a)
        left_jets.append(b)
        right_jets.append(c)
        print(i)
    return centre_lines, left_jets, right_jets
    

def graph_root_evolution(ax):
    """
    Graphs the median distance of the left and right roots from the y-axis for
    a droplet on a pool with a regression line of a+bt^{1/2}. Frames hard coded
    for a specific simulation
    
    Inputs
    -------
    ax: mpl.axes._axes.Axes
        Axes to plot graph on
    """
    
    # Get the roots
    left = RF.get_left_roots()
    right = RF.get_right_roots()
    # Set up lists to store distances in
    left_locs = []
    right_locs = []
    # For each frame, find the median distance of the left and right roots
    # from the y-axis
    for i in range(37,73):
        temp_l = np.array(left[left['time']==i].reset_index(drop=True))[:,1:4]
        temp_r = np.array(right[right['time']==i].reset_index(drop=True))[:,1:4]
        left_locs.append(np.median(np.sqrt(temp_l[:,0]**2+temp_l[:,2]**2)))
        right_locs.append(np.median(np.sqrt(temp_r[:,0]**2+temp_r[:,2]**2)))
    # Plot the left and right root distances over time
    ax.plot(np.array(range(37,73))/100,left_locs,label = 'Left Radius')
    ax.plot(np.array(range(37,73))/100,right_locs, label = 'Right Radius')
    
    # Find regression lines
    A1 = np.hstack((np.ones(len(left_locs)).reshape(len(left_locs),1),np.sqrt(np.array(range(37,73))/100).reshape(len(left_locs),1)))
    x1 = np.linalg.lstsq(A1,left_locs)[0]
    A2 = np.hstack((np.ones(len(right_locs)).reshape(len(right_locs),1),np.sqrt(np.array(range(37,73))/100).reshape(len(right_locs),1)))
    x2 = np.linalg.lstsq(A2,right_locs)[0]
    xs = np.linspace(0.37,0.72)
    left_reg = lambda y: x1[0] + x1[1]*np.sqrt(y)
    right_reg = lambda y: x2[0] + x2[1]*np.sqrt(y)
    x1[0] = '%.3f' % x1[0]
    x1[1] = '%.3f' % x1[1]
    x2[0] = '%.3f' % x2[0]
    x2[1] = '%.3f' % x2[1]
    
    # Plot regression lines
    ax.plot(xs,left_reg(xs), label = f'Left Regression, y={x1[0]}$+${x1[1]}$\sqrt{{t}}$',linestyle='dashed')
    ax.plot(xs,right_reg(xs), label = f'Right Regression, y={x2[0]}$+${x2[1]}$\sqrt{{t}}$',linestyle='dashed')
    print('Left: ',np.corrcoef(left_locs,np.sqrt(np.array(range(37,73))/100)))
    print('Right: ',np.corrcoef(right_locs,np.sqrt(np.array(range(37,73))/100)))
    ax.legend()
 
def graph_root_evolution2(ax):
    """
    Graphs the median distance of the top roots from the y-axis for
    two droplets colliding with a regression line of a+bt^{1/2}.
    Frames hard coded for a specific simulation
    
    Inputs
    -------
    ax: mpl.axes._axes.Axes
        Axes to plot graph on
    """
    
    # Get the roots
    left = RF.get_left_roots2()
    # Set up lists to store distances in
    left_locs = []
    # For each frame, find the median distance of the top roots
    # from the y-axis
    for i in range(27,65):
        temp_l = np.array(left[left['time']==i].reset_index(drop=True))[:,1:4]
        left_locs.append(np.median(np.sqrt(temp_l[:,0]**2+temp_l[:,2]**2)))
    # Plot the top root distances over time
    ax.plot(np.array(range(27,65))/100,left_locs,label = 'Top Radius')
    
    # Find regression line
    A1 = np.hstack((np.ones(len(left_locs)).reshape(len(left_locs),1),np.sqrt(np.array(range(27,65))/100).reshape(len(left_locs),1)))
    x1 = np.linalg.lstsq(A1,left_locs)[0]
    xs = np.linspace(0.27,0.64)
    left_reg = lambda y: x1[0] + x1[1]*np.sqrt(y)
    x1[0] = '%.3f' % x1[0]
    x1[1] = '%.3f' % x1[1]
    
    # Plot regression line
    ax.plot(xs,left_reg(xs), label = f'Regression, y={x1[0]}$+${x1[1]}$\sqrt{{t}}$',linestyle='dashed')
    print('Top: ',np.corrcoef(left_locs,np.sqrt(np.array(range(27,65))/100),rowvar=True))
    ax.legend()
       
def fit_ellipse(x, y):
    """
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.
    
    Inputs
    -------
    x: np.ndarray
        x-coordinates of points
    y: np.ndarray
        y-coordinates of points
        
    Returns
    -------
    params: np.ndarray, shape = (1,6)
        Array containing the coefficients a,b,c,d,e and f
        
    Code taken from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    eigval = np.real_if_close(eigval)
    eigvec = np.real_if_close(eigvec)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def jet_thickness(centre_line,left_jet,right_jet,ax=None):
    """
    Method 1 for finding the thickness of the jet, see report for details.
    This algorithm worked best for the droplets colliding
    
    Inputs
    -------
    centre_line: np.ndarray
        Coordinates of the points along the centre line
    left_jet: np.ndarray
        Coordinates of the points along the left/top side of the jet
    right_jet: np.ndarray
        Coordinates of the points along the right/bottom side of the jet
    ax: mpl.axes._axes.Axes
        (optional) axes to plot and demonstrate the thickness on
    
    Returns
    -------
    thickness: np.ndarray
        Array containing the thickness values along the jet
    """
    
    # Create a singular array for the jet
    jet = np.vstack((left_jet,right_jet))
    
    # If axes provided, plot the jet and the centre line
    if ax is not None:
        ax.scatter(np.sqrt(jet[:,0]**2+jet[:,2]**2),jet[:,1],marker='.')
        ax.plot(np.sqrt(centre_line[:,0]**2+centre_line[:,2]**2),centre_line[:,1],color='C1')
    
    # Exclude 1st and 5th quintiles of centre line data
    n = len(centre_line)
    lb = int(0.2*n)
    rb = int(0.8*n)
    # Make array of x* coordinates rather than x and z
    points = np.sqrt(centre_line[:,0]**2+centre_line[:,2]**2)
    points = points.reshape((len(points),1))
    points = np.hstack((points,centre_line[:,1].reshape((len(points),1))))
    thickness = np.zeros(rb-lb)
    # For each point along the centre line, calculate the thickness of the jet
    # as described in the report
    for i in range(lb,rb):
        point = points[i]
        num = 0
        denom = 0
        for j in range(i-2,i+3):
            num += (points[j][0]-point[0])*(points[j][1]-point[1])
            denom += (points[j][0]-point[0])**2
        grad = num/denom
        C = grad*point[0] - point[1]
        left_point = left_jet[i]
        left_x = np.sqrt(left_point[0]**2+left_point[2]**2)
        lt = point_to_line_dist([left_x,left_point[1]],-grad,1,C)
        right_point = right_jet[i]
        right_x = np.sqrt(right_point[0]**2+right_point[2]**2)
        rt = point_to_line_dist([right_x,right_point[1]],-grad,1,C)
        thickness[i-lb] = lt+rt
        dx = thickness[i-lb]/np.sqrt(1+1/grad**2)
        
        # If axes provided, plot the thickness line
        if ax is not None:
            xs = np.linspace(point[0]-dx*lt/(lt+rt),point[0]+dx*rt/(lt+rt))
            normal = lambda x: -1/grad*(x-point[0])+point[1]
            ax.plot(xs,normal(xs),color='C2');
    # Smooth outliers as described in report
    N = len(thickness)
    for k in range(int(N/2),N-1):
        if abs((thickness[k]-thickness[k-1])/thickness[k-1])>0.1:
            thickness[k] = (thickness[k+1]+thickness[k-1])/2
    for k in reversed(range(1,int(N/2))):
        if abs((thickness[k]-thickness[k+1])/thickness[k+1])>0.1:
            thickness[k] = (thickness[k+1]+thickness[k-1])/2
    return thickness

def point_to_line_dist(point,A,B,C):
    """
    Finds the perpendicular distance of a point to a line in 2D
    Line must come in the form Ax + By + C = 0
    
    Inputs
    -------
    point: np.ndarray
        Point to measure distance to
    A: float
        x coefficient
    B: float
        y coefficient
    C: float
        Constant
    
    Returns
    -------
    dist: float
        Perpendicular istance from the point to the line
    """
    # Formula as stated in report
    num = abs(A*point[0] + B*point[1] + C)
    denom = np.sqrt(A**2 + B**2)
    return num/denom

def jet_width(centre_line,left_jet,right_jet,ax=None):
    """
    Method 2 for finding the thickness of the jet, see report for details.
    This algorithm worked best for the droplet on a pool. Named width in order
    to distinguish from the thickness
    
    Inputs
    -------
    centre_line: np.ndarray
        Coordinates of the points along the centre line
    left_jet: np.ndarray
        Coordinates of the points along the left/top side of the jet
    right_jet: np.ndarray
        Coordinates of the points along the right/bottom side of the jet
    ax: mpl.axes._axes.Axes
        (optional) axes to plot and demonstrate the thickness on
    
    Returns
    -------
    width: np.ndarray
        Array containing the thickness values along the jet
    """
    
    # Create a singular array for the jet
    jet = np.vstack((left_jet,right_jet))
    
    # If axes provided, plot the jet and the centre line
    if ax is not None:
        ax.scatter(np.sqrt(jet[:,0]**2+jet[:,2]**2),jet[:,1],marker='.')
        ax.plot(np.sqrt(centre_line[:,0]**2+centre_line[:,2]**2),centre_line[:,1],color='C1')
        
    # Exclude 1st and 5th quintiles of centre line data
    n = len(centre_line)
    lb = int(0.2*n)
    rb = int(0.8*n)
    # Make arrays of x* coordinates rather than x and z
    jet = np.hstack((np.sqrt(jet[:,0]**2+jet[:,2]**2
                            ).reshape((len(jet),1)),jet[:,1].reshape((len(jet),1))))
    points = np.hstack((np.sqrt(centre_line[:,0]**2+centre_line[:,2]**2
                               ).reshape((len(centre_line),1)),centre_line[:,1]
                        .reshape((len(centre_line)),1)))
    widths = []
    # For each point along the centre line, calculate the thickness of the jet
    # as described in the report
    for i in range(lb,rb):
        point = points[i]
        num = 0
        denom = 0
        for j in range(i-2,i+3):
                num += (points[j][0]-point[0])*(points[j][1]-point[1])
                denom += (points[j][0]-point[0])**2
        grad = -denom/num
        A = -grad
        B = 1
        C = grad*point[0] - point[1]
        dists = []
        for i, jet_point in enumerate(jet):
            dists.append([point_to_line_dist(jet_point,A,B,C),i])
        dists = np.array(dists)
        dists = dists[dists[:,0].argsort()]
        coord1 = jet[int(dists[0][1])]
        coord2 = jet[int(dists[1][1])]
        if np.linalg.norm(coord1-point) > np.linalg.norm(coord2-point):
            coord1 = jet[int(dists[1][1])]
            coord2 = jet[int(dists[0][1])]
        n = 2
        while np.linalg.norm(coord1-coord2) < np.linalg.norm(coord1-point):
            coord2 = jet[int(dists[n][1])]
            n += 1
        widths.append(np.linalg.norm(coord1-coord2))
        if ax is not None:
            xs = np.linspace(coord1[0],coord2[0])
            line = lambda x: (coord1[1]-coord2[1])/(coord1[0]-coord2[0])*(x-coord1[0])+coord1[1]
            ax.plot(xs,line(xs),color='C2')
        
    # Smooth outliers as described in report
    N = len(widths)
    for k in range(int(N/2),N-1):
        if abs((widths[k]-widths[k-1])/widths[k-1])>0.1:
            widths[k] = (widths[k+1]+widths[k-1])/2
    for k in reversed(range(1,int(N/2))):
        if abs((widths[k]-widths[k+1])/widths[k+1])>0.1:
            widths[k] = (widths[k+1]+widths[k-1])/2
    k = N-1
    if abs((widths[k]-widths[k-1])/widths[k-1])>0.1:
        widths[k] = widths[k-1]/widths[k-2] * widths[k-1]
    if abs((widths[0]-widths[1])/widths[1])>0.1:
        widths[0] = widths[1]/widths[2] * widths[1]
    return widths