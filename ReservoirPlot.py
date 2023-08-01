# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:01:06 2021

@author: David Fox
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



def plot_long_pred(u, v, ulabel, vlabel, t_start=0, t_end=40000, xlim=30, ylim=30, zlim=50, line_type="-", alpha=0.7):
    fig = plt.figure(figsize=(15,9))
    ax = fig.gca(projection="3d")
    t_start = int(t_start)
    t_end = int(t_end)
    ax.plot(v[t_start:t_end-1, 0], v[t_start:t_end-1, 1], v[t_start:t_end-1, 2], line_type, color=(0.83,0.13,0.18),
            lw=1.0, label=vlabel, alpha=alpha)
    ax.plot(u[t_start:t_end-1, 0], u[t_start:t_end-1, 1], u[t_start:t_end-1, 2], line_type, color=(0.3,0.3,0.5),
            lw=1.0, label=ulabel, alpha=alpha)
    ax.set_xlim3d(-xlim, xlim)
    ax.set_ylim3d(-ylim, ylim)
    ax.set_zlim3d(0,zlim)
    plt.legend()
    
    
def plot_short_pred(u, v, u_label, v_label, t_points, start, iters):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11,7))
    errors = [np.linalg.norm(u[i] - v[i]) for i in range(iters)]
    ax1.plot(t_points[start:iters] , u[start:iters,0], "b-", label=u_label)
    ax1.plot(t_points[start:iters] , v[start:iters,0], "r-", label=v_label)
    ax2.plot(t_points[start:iters] , u[start:iters,1], "b-")
    ax2.plot(t_points[start:iters] , v[start:iters,1], "r-")
    ax3.plot(t_points[start:iters] , u[start:iters,2], "b-")
    ax3.plot(t_points[start:iters] , v[start:iters,2], "r-")
    ax4.plot(t_points[:iters] , errors[:iters], "g-", label="Absolute Error")
    
    ax3.set_xlabel("t")
    ax4.set_xlabel("t")
    
    ax1.set_ylabel("x")
    ax2.set_ylabel("y")
    ax3.set_ylabel("z")
    ax4.set_ylabel("Absolute Error")
    
    ax1.legend()
    ax4.legend()
    

def plot_pred(u, v, ulabel, vlabel, t_points, t_start=0, t_end=80000, xlim=15, ylim=15, zlim=20,
              start=0, iters=5000, line_type='-', alpha=0.5, store=False, filename=None):

    fig = plt.figure(figsize=(13,8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    t_start = int(t_start)
    t_end = int(t_end)
    
    ax.plot(u[t_start:t_end-1, 0], u[t_start:t_end-1, 1], u[t_start:t_end-1, 2], line_type, color='b',
            lw=1.0, label=ulabel, alpha=alpha)
    ax.plot(v[t_start:t_end-1, 0], v[t_start:t_end-1, 1], v[t_start:t_end-1, 2], line_type, color='r',
            lw=1.0, label=vlabel, alpha=alpha)
    
    ax.set_xlim3d(-xlim, xlim)
    ax.set_ylim3d(-ylim, ylim)
    ax.set_zlim3d(0,zlim)
    plt.legend()

    ax1 = fig.add_subplot(4, 5, 9)
    ax2 = fig.add_subplot(4, 5, 10)
    ax3 = fig.add_subplot(4, 5, 14)
    ax4 = fig.add_subplot(4, 5, 15)

    errors = [np.linalg.norm(u[i] - v[i]) for i in range(iters)]
    ax1.plot(t_points[start:iters] , u[start:iters,0], "b-")
    ax1.plot(t_points[start:iters] , v[start:iters,0], "r-")
    ax2.plot(t_points[start:iters] , u[start:iters,1], "b-")
    ax2.plot(t_points[start:iters] , v[start:iters,1], "r-")
    ax3.plot(t_points[start:iters] , u[start:iters,2], "b-")
    ax3.plot(t_points[start:iters] , v[start:iters,2], "r-")
    ax4.plot(t_points[:iters] , errors[:iters], "g-")

    ax3.set_xlabel("t")
    ax4.set_xlabel("t")
    ax1.set_ylabel("x")
    ax2.set_ylabel("y")
    ax3.set_ylabel("z")
    ax4.set_ylabel("Absolute Error")
    
    ax1.set_ylim(-13, 13)
    ax2.set_ylim(-13, 13)
    ax4.set_ylim(0, 15)
    
    ax1.set_yticks([-10, 0, 10])
    ax2.set_yticks([-10, 0, 10])
    ax3.set_yticks([0, 10, 20])
    ax4.set_yticks([0, 5, 10, 15])
    ax.set_xticks([-15, 0, 15])
    ax.set_xticks(np.arange(-15, 15, 5), minor=True)
    ax.set_yticks([-15, 0, 15])
    ax.set_yticks(np.arange(-15, 15, 5), minor=True)
    ax.set_zticks([0, 10, 20])
    ax.set_zticks(np.arange(0, 20, 5), minor=True)
    
    plt.subplots_adjust(wspace=0.5)
    
    if store:
        fn = Path(filename + '.svg').expanduser()
        fig.savefig(fn, bbox_inches='tight')

       
def plot_att(A, param_space, i=0, t_start=0, t_end=40000, xlim=300, ylim=300, zlim=300):
    u = A[param_space[i]]
    fig = plt.figure(figsize=(10,6))
    ax = fig.gca(projection="3d")
    t_start = int(t_start)
    t_end = int(t_end)
    ax.plot(u[t_start:t_end, 0], u[t_start:t_end, 1], u[t_start:t_end, 2], ".", color = (0.83,0.13,0.18),
            markersize=4, label='Predicted orbit: rho = ' + str(param_space[i]))
    
    ax.set_xlim3d(-xlim, xlim)
    ax.set_ylim3d(-ylim, ylim)
    ax.set_zlim3d(0,zlim)
    print(u[-5:-1])
    plt.legend()
        
    
def plot_traj(traj, t_start=0, t_end=40000, xlim=300, ylim=300, zlim=300):
    fig = plt.figure(figsize=(15,9))
    ax = fig.gca(projection="3d")
    t_start = int(t_start)
    t_end = int(t_end)
    for i in range(len(traj)-1):
        u = traj[i]
        ax.plot(u[t_start:t_end, 0], u[t_start:t_end, 1], u[t_start:t_end, 2], "-",
                markersize=2.0, alpha=0.7)
    u = traj[len(traj)-1]
    ax.plot(u[t_start:t_end, 0], u[t_start:t_end, 1], u[t_start:t_end, 2], "-",
            markersize=2.0, alpha=0.7, label="Reservoir Prediction")
    
    ax.set_xlim3d(-xlim, xlim)
    ax.set_ylim3d(-ylim, ylim)
    ax.set_zlim3d(0,zlim)
    plt.legend()