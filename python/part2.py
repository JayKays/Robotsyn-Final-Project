
import numpy as np
from matplotlib import pyplot as plt
from cv2 import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from HW5 import *
from util import *

def BF_matching(img1,img2, threshold = 0.75):

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append(m)
    
    print(f"Found {len(good)} matches with distance threshold = {threshold}")
    # cv.drawMatchesKnn expects list of lists as matches.


    p1 = np.array([kp1[m.queryIdx].pt for m in good])
    p2 = np.array([kp2[m.trainIdx].pt for m in good])

    uv1 = np.vstack((p1.T, np.ones(p1.shape[0])))
    uv2 = np.vstack((p2.T, np.ones(p2.shape[0])))

    np.savetxt("uv1.txt", uv1)
    np.savetxt("uv2.txt", uv2)

    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = cv.DrawMatchesFlags_DEFAULT)

    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3), plt.show()

    # return np.array(points1), np.array(points2)

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

    # return np.array(points1), np.array(points2)

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

def pose(angles, translation, R0):
    ''' Calculates the pose from parametrization'''

    s, c = np.sin, np.cos

    Rx = lambda a: np.array([[1, 0, 0], [0, c(a), s(a)], [0, -s(a), c(a)]])
    Ry = lambda a: np.array([[c(a), 0, -s(a)], [0, 1, 0], [s(a), 0, c(a)]])
    Rz = lambda a: np.array([[c(a), s(a), 0], [-s(a), c(a), 0], [0, 0, 1]])

    R = Rx(angles[0]) @ Ry(angles[1]) @ Rz(angles[2]) @ R0
    t = translation

    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t

    return T

def residual(p, R0, K, uv1, uv2):
    n_points= uv1.shape[1]

    X = np.hstack((np.reshape(p[6:], (n_points, 3)), np.ones((n_points,1))))
    T = pose(p[:3], p[3:6], R0)

    uv1_hat = project(K, X.T)
    uv2_hat = project(K, T @ X.T)
    
    r = np.hstack((uv1_hat - uv1, uv2_hat - uv2))

    return np.ravel(r.T)

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

def do_stuff():
    img1 = cv.imread("../hw5_data_ext/IMG_8207.jpg", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../hw5_data_ext/IMG_8228.jpg", cv.IMREAD_GRAYSCALE)
    K = np.loadtxt("../hw5_data_ext/K.txt")

    # img1 = cv.imread("../Rubik/IMG_3949.JPEG", cv.IMREAD_GRAYSCALE)
    # img2 = cv.imread("../Rubik/IMG_3952.JPEG", cv.IMREAD_GRAYSCALE)
    # K = np.load("mtx.csv.npy")
    
    #Matching
    FLANN_matching(img1, img2)

    uv1 = np.loadtxt('uv1.txt')
    uv2 = np.loadtxt('uv2.txt')
    xy1 = np.linalg.inv(K) @ uv1
    xy2 = np.linalg.inv(K) @ uv2

    #Estimating E
    E, inliers = estimate_E_ransac(xy1, xy2, K)
    np.savetxt('inliers.txt', inliers)
    np.savetxt('E.txt', E)

    inliers = np.loadtxt('inliers.txt').astype(bool)
    E = np.loadtxt('E.txt')

    #Extracting inlier set
    xy1 = xy1[:,inliers]
    xy2 = xy2[:,inliers]
    uv1 = uv1[:,inliers]
    uv2 = uv2[:,inliers]

    E = estimate_E(xy1, xy2)

    #Extracting pose from E
    T4 = decompose_E(E)
    T, X = choose_pose(T4, xy1, xy2) 


    #Least squares bundle adjustment
    R0 = T[:3,:3]
    t0 = T[:3,-1]
    p0 = np.hstack(([0,0,0], t0, np.ravel(X[:3,:].T)))
    print(p0.shape)

    res_func = lambda p: residual(p, R0, K, uv1[:2,:], uv2[:2,:])
    print(res_func(p0).shape)

    sparsity = bundle_adjustment_sparsity(X.shape[1])
    print(sparsity.toarray().shape)
    res = least_squares(res_func, p0, verbose=2, jac_sparsity=sparsity, x_scale='jac')
    p_opt = res['x']

    #Extracting camera pose and 3d points from bundle adjustment
    n_points= uv1.shape[1]
    T_opt = pose(p_opt[:3], p_opt[3:6], R0)
    X_opt = np.hstack((np.reshape(p_opt[6:], (n_points, 3)), np.ones((n_points,1)))).T

    np.savetxt('3D_points', X_opt)

    #Plotting results
    img1 = plt.imread("../hw5_data_ext/IMG_8207.jpg")/255.
    img2 = plt.imread("../hw5_data_ext/IMG_8228.jpg")/255.
    # img1 = cv.imread("../Rubik/IMG_3949.JPEG")/255.
    # img2 = cv.imread("../Rubik/IMG_3952.JPEG")/255.
    np.random.seed(123) # Comment out to get a random selection each time
    draw_correspondences(img1, img2, uv1, uv2, F_from_E(E, K), sample_size=8)
    draw_point_cloud(X_opt, img1, uv1, xlim=[-1.5,+1.5], ylim=[-1.5,+1.5], zlim=[0.5, 3.5], name = 'Optimal')
    draw_point_cloud(X, img1, uv1, xlim=[-1.5,+1.5], ylim=[-1.5,+1.5], zlim=[0.5, 3.5])

    plt.show()

def test():
    
    img1 = cv.imread("../hw5_data_ext/IMG_8207.jpg", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../hw5_data_ext/IMG_8228.jpg", cv.IMREAD_GRAYSCALE)
    K = np.loadtxt("../hw5_data_ext/K.txt")

    uv1 = np.loadtxt('uv1.txt')
    uv2 = np.loadtxt('uv2.txt')
    xy1 = np.linalg.inv(K) @ uv1
    xy2 = np.linalg.inv(K) @ uv2

    inliers = np.loadtxt('inliers.txt').astype(bool)
    E = np.loadtxt('E.txt')

    #Extracting inlier set
    xy1 = xy1[:,inliers]
    xy2 = xy2[:,inliers]
    uv1 = uv1[:,inliers]
    uv2 = uv2[:,inliers]

    E = estimate_E(xy1, xy2)

    #Extracting pose from E
    T4 = decompose_E(E)
    T, X = choose_pose(T4, xy1, xy2) 

    #Least squares bundle adjustment
    R0 = T[:3,:3]
    t0 = T[:3,-1]

    n_test_points = 3
    p0 = np.hstack(([0,0,0], t0, np.ravel(X[:3,:n_test_points].T)))
    print(p0)

    sparsity = bundle_adjustment_sparsity(n_test_points)
    res_func = lambda p: residual(p, R0, K, uv1[:2,:n_test_points], uv2[:2,:n_test_points])
    res = least_squares(res_func, p0, jac_sparsity=sparsity, verbose=2, x_scale='jac')
    x_opt = res.x
    jac_opt = res.jac.toarray()

    #Plot structure of different jacobians and sparsity
    test_opt = jac_opt.astype(bool).astype(int)
    test_jac = jacobian(res_func, p0, 1e-5).astype(bool).astype(int)
    test_spars = bundle_adjustment_sparsity(n_test_points).toarray()

    plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(test_opt)
    plt.title("LS output jac")
    plt.subplot(1,3,2)
    plt.imshow(test_jac)
    plt.title("Test")
    plt.subplot(1,3,3)
    plt.imshow(test_spars)
    plt.title("Sparsity")
    plt.show()


    # print(test_jac)
    # print(test_spars)

if __name__ == "__main__":

    # do_stuff()
    test()