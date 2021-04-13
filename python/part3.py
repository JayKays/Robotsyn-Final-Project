import numpy as np
from matplotlib import pyplot as plt
from cv2 import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from HW5 import *

def choose_pose(poses, xy1, xy2):
    best_num_visible = 0
    for i, T in enumerate(poses):
        P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        P2 = T[:3,:]
        X1 = triangulate_many(xy1, xy2, P1, P2)
        X2 = T@X1
        num_visible = np.sum((X1[2,:] > 0) & (X2[2,:] > 0))
        if num_visible > best_num_visible:
            best_num_visible = num_visible
            best_T = T
            best_X1 = X1
    return best_T, best_X1

def calculate_pose_points(K, matches):
    uv1 = np.vstack([matches[:,:2].T, np.ones(matches.shape[0])])
    uv2 = np.vstack([matches[:,2:4].T, np.ones(matches.shape[0])])
    xy1 = np.linalg.inv(K) @ uv1
    xy2 = np.linalg.inv(K) @ uv2

    confidence = 0.99
    inlier_fraction = 0.50
    distance_threshold = 4.0
    num_trials = get_num_ransac_trials(8, confidence, inlier_fraction)
    print('Running RANSAC: %d trials, %g pixel threshold' % (num_trials, distance_threshold))
    E,inliers = estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials)
    uv1 = uv1[:,inliers]
    uv2 = uv2[:,inliers]
    xy1 = xy1[:,inliers]
    xy2 = xy2[:,inliers]

    E = estimate_E(xy1, xy2)
    poses = decompose_E(E)

    T, X = choose_pose(poses, xy1, xy2)
    return T, X, uv1, uv2

def localize(query_image, des_model, kp_model, threshold = 0.75):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp_query, des_query = sift.detectAndCompute(query_image, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des_model,des_query,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    good = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < threshold*n.distance:
            matchesMask[i]=[1,0]
            good.append(m)
    
    final_matches = []
    for match in good:
        p1 = kp_model[match.queryIdx].pt
        p2 = kp_query[match.trainIdx].pt
        final_matches.append([p1[0],p1[1],p2[0],p2[1]])
    
    K = np.loadtxt("cam_matrix.txt")
    final_matches = np.array(final_matches)
    print(final_matches.shape)

    T, X, uv1, uv2 = calculate_pose_points(K, final_matches)
    _, rvec, tvec, inliers = cv.solvePnPRansac(X.T[:,:3], uv2.T[:,:2], K, np.zeros(4))
    R,_ = cv.Rodrigues(rvec)

    return R, tvec, X, inliers, uv2

def FLANN_matching(img1, img2, threshold = 0.75):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    good = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < threshold*n.distance:
            matchesMask[i]=[1,0]
            good.append(m)
    
    final_matches = []
    match_indexes = np.zeros(len(kp1), dtype = bool)
    for match in good:
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt
        final_matches.append([p1[0],p1[1],p2[0],p2[1]])
        match_indexes[match.queryIdx] = True
    
    des_model = np.array(des1)[match_indexes]
    kp_model = np.array(kp1)[match_indexes]
    
    print(f"Found {len(good)} matches with distance threshold = {threshold}")

    p1 = np.array([kp1[m.queryIdx].pt for m in good])
    p2 = np.array([kp2[m.trainIdx].pt for m in good])

    uv1 = np.vstack((p1.T, np.ones(p1.shape[0])))
    uv2 = np.vstack((p2.T, np.ones(p2.shape[0])))

    np.savetxt("uv1.txt", uv1)
    np.savetxt("uv2.txt", uv2)

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)

    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()

    return des_model, kp_model

def visualize(I):
    """ returns everything needed for plotting"""
    img1 = cv.imread("../hw5_data_ext/IMG_8207.jpg", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../hw5_data_ext/IMG_8228.jpg", cv.IMREAD_GRAYSCALE)

    des_model, kp_model = FLANN_matching(img1, img2)
    R, tvec, X, inliers, uv2 = localize(I, des_model, kp_model)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,-1] = tvec[:,0]

    return X, T, inliers, uv2

if __name__ == "__main__":
    img1 = cv.imread("../hw5_data_ext/IMG_8207.jpg", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../hw5_data_ext/IMG_8228.jpg", cv.IMREAD_GRAYSCALE)
    img3 = cv.imread("../hw5_data_ext/IMG_8229.jpg", cv.IMREAD_GRAYSCALE)
    des_model, kp_model = FLANN_matching(img1, img2)
    R, tvec, X, inliers, uv2 = localize(img3, des_model, kp_model)