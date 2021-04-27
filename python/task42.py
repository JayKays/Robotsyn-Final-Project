import numpy as np
from cv2 import cv2 as cv

from HW5 import *


def FLANN_matching(img1, img2, threshold = 0.75):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

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
    
    print(f"Found {len(good)} matches with distance threshold = {threshold}")

    p1 = np.array([kp1[m.queryIdx].pt for m in good])
    p2 = np.array([kp2[m.trainIdx].pt for m in good])

    des = des1[[m.queryIdx for m in good], :]

    uv1 = np.vstack((p1.T, np.ones(p1.shape[0])))
    uv2 = np.vstack((p2.T, np.ones(p2.shape[0])))

    # np.savetxt("uv1.txt", uv1)
    # np.savetxt("uv2.txt", uv2)

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)

    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    # plt.imshow(img3,),plt.show()

    return p1, p2, des

def bundle_adjustment_sparsity(n_points, n_cameras = 2):
    m = 2*n_points * n_cameras
    n = 6*(n_cameras-1) + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    for i in range(n_cameras-1):
        A[2*(i+1)*n_points:2*(i+2)*n_points, 6*i:6*(i+1)] = 1
        
    for s in range(n_points):
        for i in range(n_cameras):
            A[s*2 + i*2*n_points:(s+1)*2 + i*2*n_points, s*3 + 6*(n_cameras-1):(s+1)*3 + 6*(n_cameras-1)] = 1

    return A

def match_multi_images(images, K, threshold = 0.75):
    points = []
    des = []
    poses = [np.eye(4)]

    n = len(images)
    for i in range(1):
        for j in range(i+1, n):
            p1, p2, point_des = FLANN_matching(images[i], images[j])

            if i == 0:
                X, point_des, pose = model_points_from_match(p1, p2, point_des, K)
                poses.append(pose)
            else:
                X, point_des, _ = model_points_from_match(p1, p2, point_des, K, pose = poses[i])

            points.append(X)
            des.append(point_des)
    
    return np.hstack([p for p in points]), np.vstack([d for d in des])

def model_points_from_match(p1, p2, des, K, pose = None):
    # print(p2.shape)
    uv1 = np.vstack((p1.T, np.ones(p1.shape[0])))
    uv2 = np.vstack((p2.T, np.ones(p2.shape[0])))
    # print(uv1.shape)
    xy1 = np.linalg.inv(K) @ uv1
    xy2 = np.linalg.inv(K) @ uv2

    #Estimating E
    E, inliers = estimate_E_ransac(xy1, xy2, K)

    #Extracting inlier set
    xy1 = xy1[:,inliers]
    xy2 = xy2[:,inliers]
    uv1 = uv1[:,inliers]
    uv2 = uv2[:,inliers]
    
    E = estimate_E(xy1, xy2)

    #Extracting pose from E
    T4 = decompose_E(E)
    T, X = choose_pose(T4, xy1, xy2) 
    
    inlier_des = des[inliers, :]

    if pose is not None:
        pose_inv = pose
        pose_inv[:3,:3] =  pose_inv[:3,:3].T
        pose_inv[:3,-1] = (-1) * pose_inv[:3,-1]
        
        X = pose_inv @ X

    return X, inlier_des, T


if __name__ == "__main__":

    img_numbs = ['07', '28', '21', '13', '09']

    images = [cv.imread(f"../hw5_data_ext/IMG_82{img_numbs[i]}.jpg") for i in range(len(img_numbs))]
    K = np.loadtxt("../hw5_data_ext/K.txt")

    X, des = match_multi_images(images, K)

    print(X.shape)
    print(des.shape)

    uv = np.vstack((project(K, X), np.ones(X.shape[1])))

    draw_point_cloud(X, images[0], uv, xlim=[-1.5,+1.5], ylim=[-1.5,+1.5], zlim=[0.5, 3.5])
    plt.show()