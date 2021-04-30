import numpy as np
from cv2 import cv2 as cv
from scipy.optimize import least_squares
from HW5 import *

def pose(p, R0):
    s, c = np.sin, np.cos
    Rx = lambda a: np.array([[1, 0, 0], [0, c(a), s(a)], [0, -s(a), c(a)]])
    Ry = lambda a: np.array([[c(a), 0, -s(a)], [0, 1, 0], [s(a), 0, c(a)]])
    Rz = lambda a: np.array([[c(a), s(a), 0], [-s(a), c(a), 0], [0, 0, 1]])

    R = Rx(p[0]) @ Ry(p[1]) @ Rz(p[2]) @ R0
    tvec = p[3:]

    T = np.eye(4)
    T[:3,:3] = R
    T[:3,-1] = tvec

    return T

def monte_carlo_pose_cov(K_bar, sig_K, p0, uv, X, R0, m = 500):

    poses = np.zeros((m,6))

    for i in range(m):

        params = np.random.normal(np.zeros(3), sig_K)
        
        K = K_bar + np.array([[params[0], 0, params[1]], [0,params[0], params[2]], [0,0,0]])

        res_fun = lambda p: np.ravel(project(K, pose(p, R0) @ X) - uv)
        res = least_squares(res_fun, p0)
        poses[i,:] = res.x

        if not (i+1) % 100: print(f"Monte Carlo Iteration: {i+1}")

    return np.cov(poses.T)

def monte_carlo_std(K, sig_K, p0, uv, X, R0, m = 500):

    cov = monte_carlo_pose_cov(K, sig_K, p0, uv, X, R0)
    std = np.sqrt(np.diagonal(cov))

    return std