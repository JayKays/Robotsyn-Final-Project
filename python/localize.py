import numpy as np
from matplotlib import pyplot as plt
from cv2 import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from HW5 import *
from part2 import *


def localize(query_img, X, model_des, K, refined = True, weighted = False):

    img_points, world_points = match_image_to_model(X, model_des, query_img)
    p , world_points, img_points, J = estimate_pose(img_points.T, world_points, K, refined, weighted)

    return p, J, world_points, img_points

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

def estimate_pose(img_points, world_points, K, refine = True, weighted = False):

    _, rvec, tvec, inliers = cv.solvePnPRansac(world_points[:3,:].T, img_points.T, K, np.zeros(4))

    world_points = world_points[:,inliers[:,0]]
    img_points = img_points[:,inliers[:,0]]

    if weighted:
        weights = calc_weights(inliers.shape[0])
    else:
        weights = None

    p = np.hstack((rvec.T[0], tvec.T[0]))

    if refine:
        p, J = refine_pose(p, world_points, img_points, K, weights)

    return p, world_points, img_points, J

def refine_pose(p0, X, uv, K, weights = None):

    if weights is None:
        res_fun = lambda p: np.ravel(project(K, pose(p) @ X) - uv)
    else:
        res_fun = lambda p: weights @ np.ravel(project(K, pose(p) @ X) - uv)

    res = least_squares(res_fun, p0, verbose=2)
    p_opt = res.x
    J = res.jac

    if weights is not None:
        J = np.linalg.inv(weights) @ J

    return p_opt , J

def calc_weights(n, sig_u = 50, sig_v = 0.1):

    sig_r = np.hstack((np.ones(n)*sig_u, np.ones(n)*sig_v))
    sig_r = np.diag(sig_r)

    L_inv = np.linalg.inv(np.linalg.cholesky(sig_r))
    
    return L_inv

def pose_std(Jac):

    sig_r = np.eye(Jac.shape[0])
    sig_p = np.linalg.inv(Jac.T @ sig_r @ Jac)

    std = np.sqrt(np.diagonal(sig_p))

    return std

def unit_convertion(pose_std):
    '''Converts radians and meters to deg and mm'''
    pose_std[:3] *= 180/np.pi
    pose_std[3:] *= 1000
    return pose_std

def pose(p):
    rvec = p[:3]
    tvec = p[3:]

    R,_ = cv.Rodrigues(rvec)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,-1] = tvec

    return T



    
