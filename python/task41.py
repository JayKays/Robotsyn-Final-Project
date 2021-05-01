
import numpy as np
from cv2 import cv2 as cv
from scipy.optimize import least_squares
from HW5 import *
from Monte_Carlo import monte_carlo_std, pose
from localize import localize, unit_convertion

def weighted_monte_carlo_pose_cov(K_bar, sig_K, p0, R0, uv, X, m = 500):

    poses = np.zeros((m,6))

    for i in range(m):

        params = np.random.normal(np.zeros(3), sig_K)
        
        K = K_bar + np.array([[params[0], 0, params[1]], [0,params[0],params[2]],[0,0,0]])

        # res_fun = lambda p: np.ravel(project(K, pose(p[:3],p[3:]) @ X) - uv)
        res_fun = lambda p: residual(p, X, uv, K, sig_K, R0)
        res = least_squares(res_fun, p0)
        poses[i,:]= res.x

        if not (i+1)%100: print(f"Monte Carlo Iteration: {i+1}")

    return np.cov(poses.T)

def weighted_monte_carlo_std(K, sig_K, p0, uv, X, R0, m = 500):

    cov = weighted_monte_carlo_pose_cov(K, sig_K, p0, R0, uv, X)
    std = np.sqrt(np.diagonal(cov))

    return std

# def pose(p):
#     rvec = p[:3]
#     tvec = p[3:]
#     R,_ = cv.Rodrigues(rvec)
#     T = np.eye(4)
#     T[:3,:3] = R
#     T[:3,-1] = tvec

#     return T

def residual(p, X, uv, K, sig_K, R0):

    n = X.shape[1]

    sig_f = sig_K[0]
    sig_cx = sig_K[1]
    sig_cy = sig_K[2]

    X_hat = pose(p, R0) @ X

    uv_hat = project(K, X_hat)

    #Weighting the residual with std
    std_u = sig_cx**2 + (X_hat[0,:] / X_hat[2,:])**2 * sig_f**2
    std_v = sig_cy**2 + (X_hat[1,:] / X_hat[2,:])**2 * sig_f**2

    sig_r = np.sqrt(np.hstack((1/std_u, 1/std_v)))

    res = sig_r * np.ravel(uv_hat - uv)

    return res

if __name__ == "__main__":
    np.random.seed(0)

    # K = np.loadtxt("../hw5_data_ext/K.txt")
    # X = np.loadtxt("../HW5_3D_model/3D_points.txt")
    # model_des = np.loadtxt("../HW5_3D_model/descriptors").astype("float32")
    # query_img = cv.imread("../hw5_data_ext/IMG_8207.jpg")
    # # X[:3,:] *= 6.2
    K = np.loadtxt("cam_matrix.txt")
    X = np.loadtxt("../3D_model/3D_points.txt")
    model_des = np.loadtxt("../3D_model/descriptors").astype("float32")
    distortion = np.loadtxt('dist.txt')
    query_img = cv.imread('../iCloud Photos/IMG_3982.JPEG')
    # dist_std = np.loadtxt('stdInt.txt')

    p0, _, X, uv, R0 = localize(query_img, X, model_des, K)
    
    std1 = monte_carlo_std(K, [50, 0.1, 0.1], p0, uv, X, R0)
    std2 = monte_carlo_std(K, [0.1, 50, 0.1], p0, uv, X, R0)
    std3 = monte_carlo_std(K, [0.1, 0.1, 50], p0, uv, X, R0)

    std1_w = weighted_monte_carlo_std(K, [50, 0.1, 0.1], p0, uv, X, R0)
    std2_w = weighted_monte_carlo_std(K, [0.1, 50, 0.1], p0, uv, X, R0)
    std3_w = weighted_monte_carlo_std(K, [0.1, 0.1, 50], p0, uv, X, R0)

    print(np.round(unit_convertion(std1), decimals = 8))
    print(np.round(unit_convertion(std2), decimals = 8))
    print(np.round(unit_convertion(std3), decimals = 8))
    print('-'*60)
    print(np.round(unit_convertion(std1_w), decimals = 8))
    print(np.round(unit_convertion(std2_w), decimals = 8))
    print(np.round(unit_convertion(std3_w), decimals = 8))



