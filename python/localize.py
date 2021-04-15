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

def refine_pose(rvec, tvec, X, uv, K):
    
    p0 = np.hstack((rvec.T[0], tvec.T[0]))

    res_fun = lambda p: np.ravel(project(K, pose(p[:3],p[3:]) @ X) - uv)

    res = least_squares(res_fun, p0, verbose=2)
    p_opt = res['x']
    T = pose(p_opt[:3], p_opt[3:])

    return T

def pose(rvec,tvec):
    R,_ = cv.Rodrigues(rvec)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,-1] = tvec

    return T

def estimate_pose(img_points, world_points, K, refine = True):

    _, rvec, tvec, inliers = cv.solvePnPRansac(world_points[:3,:].T, img_points.T, K, np.zeros(4))
    
    world_points = world_points[:,inliers[:,0]]
    img_points = img_points[:,inliers[:,0]]

    T = pose(rvec,tvec.T[0])
    if refine:
        T = refine_pose(rvec, tvec, world_points, img_points, K)


    return T, world_points, img_points

def localize(query_img, X, model_des, K, refined = True):

    img_points, world_points = match_image_to_model(X, model_des, query_img)
    T , world_points, img_points = estimate_pose(img_points.T, world_points, K, refined)

    np.savetxt("../part3_matched_points/3D.txt", world_points)
    np.savetxt("../part3_matched_points/2D.txt", img_points)

    return T

if __name__ == "__main__":
    X = np.loadtxt("../3D_model/3D_points.txt")
    model_des = np.loadtxt("../3D_model/descriptors").astype("float32")
    query_img = cv.imread("../hw5_data_ext/IMG_8220.jpg")
    K = np.loadtxt("../hw5_data_ext/K.txt")

    # img_points, world_points = match_image_to_model(X, model_des, query_img)
    # T = localize(img_points.T, world_points, K)
    T = localize(query_img, X, model_des, K)

    # print(T)