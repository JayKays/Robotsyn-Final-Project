import numpy as np
from matplotlib import pyplot as plt
from cv2 import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from HW5 import *
from part2 import *

def match_image_to_model(X, model_des, img, threshold = 0.75):

    sift = cv.SIFT_create()
    kp_query, query_des = sift.detectAndCompute(img, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(model_des, query_des , k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    good = []

    # ratio test as per Lowe's paper
    matched_idx = [False]*X.shape[1]
    for i,(m,n) in enumerate(matches):
        if m.distance < threshold*n.distance:
            matchesMask[i]=[1,0]
            good.append(m)
            matched_idx[i] = True
    
    print(f"Found {len(good)} matches with distance threshold = {threshold}")

    matched_2D_points = np.array([kp_query[m.trainIdx].pt for m in good])
    matched_3D_points = X[:,matched_idx]

    return matched_2D_points, matched_3D_points

def refine_pose(rvec, tvec, X, uv, K, weights = None):
    
    p0 = np.hstack((rvec.T[0], tvec.T[0]))
    if weights is None:
        res_fun = lambda p: np.ravel(project(K, pose(p[:3],p[3:]) @ X) - uv)
    else:
        res_fun = lambda p: weights @ np.ravel(project(K, pose(p[:3],p[3:]) @ X) - uv)

    res = least_squares(res_fun, p0, verbose=2)
    p_opt = res.x
    J = res.jac

    if weights is not None:
        J = np.linalg.inv(weights) @ J

    # print(np.round(pose_std(res.jac), decimals = 10))

    T = pose(p_opt[:3], p_opt[3:])

    return T, J

def pose(rvec,tvec):
    R,_ = cv.Rodrigues(rvec)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,-1] = tvec

    return T

def estimate_pose(img_points, world_points, K, refine = True, weighted = False):

    _, rvec, tvec, inliers = cv.solvePnPRansac(world_points[:3,:].T, img_points.T, K, np.zeros(4))

    world_points = world_points[:,inliers[:,0]]
    img_points = img_points[:,inliers[:,0]]
    if weighted:
        weights = calc_weights(inliers.shape[0])
    else:
        weights = None

    T = pose(rvec,tvec.T[0])

    if refine:
        T, J = refine_pose(rvec, tvec, world_points, img_points, K, weights)

    np.savetxt("../Monte_Carlo_init_params/X.txt", world_points)
    np.savetxt("../Monte_Carlo_init_params/uv.txt", img_points)
    np.savetxt("../Monte_Carlo_init_params/pose.txt", np.hstack((rvec.T[0],tvec.T[0])))

    return T, world_points, img_points, J

def localize(query_img, X, model_des, K, refined = True, weighted = False):

    img_points, world_points = match_image_to_model(X, model_des, query_img)
    T , world_points, img_points, J = estimate_pose(img_points.T, world_points, K, refined, weighted)

    np.savetxt("../part3_matched_points/3D.txt", world_points)
    np.savetxt("../part3_matched_points/2D.txt", img_points)



    return T, J

def sig_p(Jac):
    return np.linalg.inv(Jac.T @ np.eye(Jac.shape[0]) @ Jac)

def pose_std(Jac):
    return np.sqrt(np.diagonal(sig_p(Jac))**2)

def unit_convertion(pose_std):
    return pose_std

def calc_weights(n):
    sig_x = 50
    sig_y = 0.1

    w_inv = np.hstack((np.ones(n)*sig_x, np.ones(n)*sig_y))
    # print(np.diag(w_inv).shape)
    return np.linalg.inv(np.diag(w_inv))


def monte_carlo_pose_cov(K_bar, sig_K, m = 500):

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

def monte_carlo_std(K, sig_K, m = 500):

    cov = monte_carlo_pose_cov(K, sig_K, m)
    std = np.sqrt(np.diagonal(cov))

    return std
if __name__ == "__main__":
    X = np.loadtxt("../3D_model/3D_points.txt")
    model_des = np.loadtxt("../3D_model/descriptors").astype("float32")
    query_img = cv.imread("../hw5_data_ext/IMG_8220.jpg")
    K = np.loadtxt("../hw5_data_ext/K.txt")

    # weights = calc_weights(X.shape[1])

    # T, J = localize(query_img, X, model_des, K, weighted= True)

    # print(pose_std(J))

    T, J = localize(query_img, X, model_des, K)

    std1 = monte_carlo_std(K, [50, 0.1, 0.1])
    std2 = monte_carlo_std(K, [0.1, 50, 0.1])
    std3 = monte_carlo_std(K, [0.1, 0.1, 50])

    print(std1)
    print(std2)
    print(std3)

    # print(T)
