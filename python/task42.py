import numpy as np
import glob
from cv2 import cv2 as cv
from pathlib import Path
from matplotlib import pyplot as plt

from HW5 import *
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from visualize_query_results import visualize_query_res

from localize import localize
from util import draw_model_and_query_pose



# def FLANN_matching(kp1, kp2, des1, des2, threshold = 0.75):
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


    good = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < threshold*n.distance:
            good.append(m)
    
    print(f"Found {len(good)} matches with distance threshold = {threshold}")

    p1 = np.array([kp1[m.queryIdx].pt for m in good])
    p2 = np.array([kp2[m.trainIdx].pt for m in good])

    des = des1[[m.queryIdx for m in good], :]

    uv1 = np.vstack((p1.T, np.ones(p1.shape[0])))
    uv2 = np.vstack((p2.T, np.ones(p2.shape[0])))

    return p1, p2, des



def realtive_scale(X1, X2, des1, des2, threshold = 0.75):

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)


    good = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < threshold*n.distance:
            good.append(m)
    
    p1 = np.array([X1[:,m.queryIdx] for m in good])
    p2 = np.array([X2[:,m.trainIdx] for m in good])

    diff1 = np.linalg.norm(np.diff(p1, axis = 1), axis = 0)
    diff2 = np.linalg.norm(np.diff(p2, axis = 1), axis = 0)

    scale = np.mean(diff1 / (diff2))

    return scale


def match_multi_images(images, K, threshold = 0.75):
    points = []
    des = []

    T = np.eye(4)

    n = len(images)

    # sift = cv.SIFT_create()

    # for i in range(n):
    #     kp, des = sift.detectAndCompute(images[i], None)

    #     img_des.append(des)

    # print(len(img_des))
    #Initial model from first 2 images
    # p1, p2, point_des = FLANN_matching(images[0], images[1])
    # X, point_des, pose, p1, p2 = model_points_from_match(p1, p2, point_des, K)


    for i in range(1,n):
        # p1, p2, point_des = FLANN_matching(kp[i-1], kp[i], des[i-1], des[i])
        p1, p2, point_des = FLANN_matching(images[i-1], images[i])

        X, point_des, pose, p1, p2 = model_points_from_match(p1, p2, point_des, K)

        # print(len(points))

        if i > 1:
            scale = realtive_scale(points[i-2], X, des[i-2], point_des)
            pose[:3,-1] *= scale
            X[:3,:] *= scale
        else:
            pose[:3,-1] *= 5
            X[:3,:] *= 5

        T_opt, X_opt = bundle_adjustment(pose, X, p1, p2, K)
        
        points.append(np.linalg.inv(T) @ X_opt)

        des.append(point_des)

        T = T_opt @ T
    
    return np.hstack([p for p in points]), np.vstack([d for d in des])


def model_points_from_match(p1, p2, des, K):
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

    return X, inlier_des, T, uv1, uv2

def bundle_adjustment(T, X, p1, p2, K):
    if p1.shape[1] == 2:
        uv1 = np.vstack((p1.T, np.ones(p1.shape[0])))
        uv2 = np.vstack((p2.T, np.ones(p2.shape[0])))
    elif p1.shape[0] == 3:
        uv1 = p1
        uv2 = p2
    else:
        raise "Input points wrong dimentions"

    R0 = T[:3,:3]
    t0 = T[:3,-1]
    p0 = np.hstack(([0,0,0], t0, np.ravel(X[:3,:].T)))
    # print(p0.shape)
    n_points= uv1.shape[1]

    res_func = lambda p: residual(p, R0, K, uv1[:2,:], uv2[:2,:])
    
    sparsity = bundle_adjustment_sparsity(n_points)

    res = least_squares(res_func, p0, ftol = 1e-3,verbose = 2, jac_sparsity=sparsity, x_scale='jac')
    p_opt = res['x']
    
    #Extracting camera pose and 3d points from bundle adjustment
    T_opt = pose(p_opt[:6], R0)
    X_opt = np.hstack((np.reshape(p_opt[6:], (n_points, 3)), np.ones((n_points,1)))).T

    return T_opt, X_opt

def residual(p, R0, K, uv1, uv2):
    n_points= uv1.shape[1]

    X = np.hstack((np.reshape(p[6:], (n_points, 3)), np.ones((n_points,1))))
    T = pose(p[:6], R0)

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

def save_model(X, des, uv, path):
    np.savetxt(Path.joinpath(path,'3D_points.txt'), X)
    np.savetxt(Path.joinpath(path,'descriptors'), np.array(des))
    np.savetxt(Path.joinpath(path,'uv.txt'), uv)

def generate_model(img_numbs,save = False):

    images = [cv.imread(f"../hw5_data_ext/IMG_82{img_numbs[i]}.jpg") for i in range(len(img_numbs))]
    K = np.loadtxt("../hw5_data_ext/K.txt")

    X, des = match_multi_images(images, K)

    # print(X.shape)
    # print(des.shape)

    img = plt.imread(f"../hw5_data_ext/IMG_82{img_numbs[0]}.jpg")/255.

    uv = project(K, X).astype(np.int32)
    # print(img.shape)
    uv[0,:] = np.where(uv[0,:]< img.shape[1], uv[0,:], img.shape[1]-1)
    uv[1,:] = np.where(uv[1,:]< img.shape[0], uv[1,:], img.shape[0]-1)

    if save:
        save_model(X, des, uv, Path('../Task42_model'))

    # draw_point_cloud(X, img, uv, xlim=[-10,10], ylim=[-10,+10], zlim=[10,30], find_colors=True)
    # plt.show()


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

if __name__ == "__main__":

    img_numbs = ['07','27', '09', '10', '11']
    query_numbs = ['12', '13', '14']
    Two_view_model = False
    # generate_model(img_numbs)

    K = np.loadtxt("../hw5_data_ext/K.txt")

    if Two_view_model:
        X = np.loadtxt("../HW5_3D_model/3D_points.txt")
        model_des = np.loadtxt("../HW5_3D_model/descriptors").astype("float32")

        #Used for coloring the point cloud
        uv = np.loadtxt("../HW5_3D_model/uv.txt")
        model_img = plt.imread(f"../hw5_data_ext/IMG_82{img_numbs[0]}.jpg")/255.
        c = model_img[uv[1,:].astype(np.int32), uv[0,:].astype(np.int32), :]

        lookfrom1 = np.array((0,-20,-5))*2
        lookat1   = np.array((0,0,6))
        lookfrom2 = np.array((25,-15,-10))
        lookat2   = np.array((0,0,10))
    else:

        X = np.loadtxt("../Task42_model/3D_points.txt")
        model_des = np.loadtxt("../Task42_model/descriptors").astype("float32")
        
        #used for coloring the point cloud
        uv = np.loadtxt("../Task42_model/uv.txt")
        model_img = plt.imread(f"../hw5_data_ext/IMG_82{img_numbs[0]}.jpg")/255.
        c = model_img[uv[1,:].astype(np.int32), uv[0,:].astype(np.int32), :]

        lookfrom1 = np.array((0,-20,-5))*2
        lookat1   = np.array((0,0,6))
        lookfrom2 = np.array((25,-15,-10))
        lookat2   = np.array((0,0,10))

    query_images = [cv.imread(f"../hw5_data_ext/IMG_82{query_numbs[i]}.jpg") for i in range(len(query_numbs))]

    fig = plt.figure(figsize=(15,10))

    for i in range(len(query_images)):

        p, J, world_points, img_points, R0 = localize(query_images[i], X, model_des, K, False)

        T = pose(p, R0)
        
        #Plot estimated pose and query image
        plt.subplot(330 + 3*i + 1)
        plt.imshow(plt.imread(f"../hw5_data_ext/IMG_82{query_numbs[i]}.jpg"))

        plt.subplot(330 + 3*i + 2)
        draw_model_and_query_pose(X, T, K, lookat1, lookfrom1, c=c)
        # plt.title('Model and localized pose (top view)')

        plt.subplot(330 + 3*i + 3)
        draw_model_and_query_pose(X, T, K, lookat2, lookfrom2, c=c)
        # plt.title('Model and localized pose (side view)')

    plt.tight_layout()
    plt.show()


    # draw_point_cloud(X, model_img, uv, xlim=[-2,+2], ylim=[-2,+2], zlim=[1,6], find_colors=True)
    # # plt.show()


