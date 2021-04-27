import numpy as np
from cv2 import cv2 as cv
from scipy.optimize import least_squares
from HW5 import *


def pose(rvec,tvec):
    R,_ = cv.Rodrigues(rvec)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,-1] = tvec

    return T

def monte_carlo_pose_cov(K_bar, sig_K, p0, uv, X, m = 500):

    p0 = np.loadtxt("../Monte_Carlo_init_params/pose.txt")
    uv = np.loadtxt("../Monte_Carlo_init_params/uv.txt")
    X = np.loadtxt("../Monte_Carlo_init_params/X.txt")

    poses = np.zeros((m,6))

    for i in range(m):
        params = np.random.normal(np.zeros(3), sig_K)
        K = K_bar + np.array([[params[0], 0, params[1]], [0,params[0],params[2]],[0,0,0]])

        res_fun = lambda p: np.ravel(project(K, pose(p[:3],p[3:]) @ X) - uv)
        res = least_squares(res_fun, p0)
        poses[i,:]= res.x

        if not (i+1)%100: print(f"Monte Carlo Iteration: {i+1}")

    return np.cov(poses.T)

def monte_carlo_std(K, sig_K, p0, uv, X, m = 500):

    cov = monte_carlo_pose_cov(K, sig_K, p0, uv, X)
    std = np.sqrt(np.diagonal(cov))

    return std