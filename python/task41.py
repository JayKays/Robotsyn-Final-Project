
import numpy as np
from cv2 import cv2 as cv
from scipy.optimize import least_squares
from HW5 import *
from Monte_Carlo import monte_carlo_std
from localize import localize

def weighted_monte_carlo_pose_cov(K_bar, sig_K, p0, uv, X, m = 500):

    poses = np.zeros((m,6))

    for i in range(m):

        params = np.random.normal(np.zeros(3), sig_K)
        
        K = K_bar + np.array([[params[0], 0, params[1]], [0,params[0],params[2]],[0,0,0]])

        # res_fun = lambda p: np.ravel(project(K, pose(p[:3],p[3:]) @ X) - uv)
        res_fun = lambda p: residual(p, X, uv, K, sig_K)
        res = least_squares(res_fun, p0)
        poses[i,:]= res.x

        if not (i+1)%5: print(f"Monte Carlo Iteration: {i+1}")

    return np.cov(poses.T)

def weighted_monte_carlo_std(K, sig_K, p0, uv, X, m = 500):

    cov = weighted_monte_carlo_pose_cov(K, sig_K, p0, uv, X)
    std = np.sqrt(np.diagonal(cov))

    return std

def pose(p):
    rvec = p[:3]
    tvec = p[3:]
    R,_ = cv.Rodrigues(rvec)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,-1] = tvec

    return T

def residual(p, X, uv, K, sig_K):

    n = X.shape[1]

    sig_f = sig_K[0]
    sig_cx = sig_K[1]
    sig_cy = sig_K[2]

    X_hat = pose(p) @ X

    uv_hat = project(K, X_hat)

    std_u = sig_cx**2 + (X_hat[0,:] / X_hat[2,:])**2 * sig_f**2
    std_v = sig_cy**2 + (X_hat[1,:] / X_hat[2,:])**2 * sig_f**2

    sig_r = np.diag(np.sqrt(np.hstack((1/std_u, 1/std_v))))

    res = sig_r @ np.ravel(uv_hat - uv)

    return res

if __name__ == "__main__":
    np.random.seed(0)

    K = np.loadtxt("../hw5_data_ext/K.txt")
    X = np.loadtxt("../HW5_3D_model/3D_points.txt")
    model_des = np.loadtxt("../HW5_3D_model/descriptors").astype("float32")
    query_img = cv.imread("../hw5_data_ext/IMG_8207.jpg")
    X[:3,:] *= 6.2

    p0, _, X, uv = localize(query_img, X, model_des, K)
    
    # std1 = monte_carlo_std(K, [50, 0.1, 0.1], p0, uv, X)
    # std2 = monte_carlo_std(K, [0.1, 50, 0.1], p0, uv, X)
    # std3 = monte_carlo_std(K, [0.1, 0.1, 50], p0, uv, X)

    # std1_w = weighted_monte_carlo_std(K, [50, 0.1, 0.1], p0, uv, X)
    # std2_w = weighted_monte_carlo_std(K, [0.1, 50, 0.1], p0, uv, X)
    # std3_w = weighted_monte_carlo_std(K, [0.1, 0.1, 50], p0, uv, X)

    # print('[4.38717090e-05 4.81921459e-05 5.07619846e-06 4.75064375e-05 5.59353675e-05 3.81880963e-02]')
    # print('[0.00050253 0.00226796 0.00056173 0.005338   0.00272789 0.03975035]')
    print(np.round(pose(p0), decimals = 5))
    # print(np.round(std1, decimals = 6))
    # print(np.round(std2, decimals = 6))
    # print(np.round(std3, decimals = 6))



