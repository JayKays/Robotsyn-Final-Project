
import numpy as np
from matplotlib import pyplot as plt
from cv2 import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from HW5 import *
from util import *
from part1 import undistort_img
from visualize_query_results import *


def ORB_matching(img1, img2, threshold = 45):
    # Initiate ORB detector
    orb = cv.ORB_create(40000) #specifying nr of keypoints to locate

    image1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    image2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(image1_gray, None)
    kp2, des2 = orb.detectAndCompute(image2_gray, None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)
    print(len(matches))

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    good = []

    # ratio test as per Lowe's paper
    for m in matches[:2000]:
        # if m.distance < threshold:
        good.append(m)

    p1 = np.array([kp1[m.queryIdx].pt for m in good])
    p2 = np.array([kp2[m.trainIdx].pt for m in good])

    des = des1[[m.queryIdx for m in good], :]

    uv1 = np.vstack((p1.T, np.ones(p1.shape[0])))
    uv2 = np.vstack((p2.T, np.ones(p2.shape[0])))

    np.savetxt("uv1.txt", uv1)
    np.savetxt("uv2.txt", uv2)

    print(f"Found {len(good)} matches with distance threshold = {threshold}")

    # draw first 2000 matches
    img3 = cv.drawMatches(image1_gray, kp1, image2_gray, kp2, matches[:2000], image2_gray, flags = 2)
    plt.imshow(img3),plt.show()

    return p1, p2, des

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
    
def generate_model(p1, p2, K, des):
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
    # print(np.array(des).shape)
    inlier_des = des[inliers, :]
    # print(np.array(inlier_des).shape)

    return X, inlier_des, T, uv1, uv2, E
    
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

    res_func = lambda p: residual(p, R0, K, uv1[:2,:], uv2[:2,:])
    print(np.mean(np.sqrt(res_func(p0)**2)))
    sparsity = bundle_adjustment_sparsity(X.shape[1])

    res = least_squares(res_func, p0, verbose=2, jac_sparsity=sparsity, x_scale='jac')
    p_opt = res['x']
    print(np.mean(np.sqrt(res_func(p_opt)**2)))
    #Extracting camera pose and 3d points from bundle adjustment
    n_points= uv1.shape[1]
    T_opt = pose(p_opt[:3], p_opt[3:6], R0)
    X_opt = np.hstack((np.reshape(p_opt[6:], (n_points, 3)), np.ones((n_points,1)))).T

    return T_opt, X_opt
    
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

def plot_point_cloud(X, uv, img):
    draw_point_cloud(X, img, uv, xlim=[-1.5,+1.5], ylim=[-1.5,+1.5], zlim=[0.5, 3.5])
    # draw_point_cloud(X, img1, uv1, xlim=[-1.5,+1.5], ylim=[-1.5,+1.5], zlim=[0.5, 3.5])

def save_model(X, des):
    np.savetxt("../3D_model/3D_points.txt", X)
    np.savetxt("../3D_model/descriptors", np.array(des))

if __name__ == "__main__":

    img1 = cv.imread('../iCloud Photos/IMG_3980.JPEG')
    img2 = cv.imread('../iCloud Photos/IMG_3981.JPEG')

    K = np.loadtxt("cam_matrix.txt")
    dist = np.loadtxt('dist.txt')
    stdInt = np.loadtxt('stdInt.txt')

    # img1 = undistort_img(img1, K, dist, stdInt)
    # img2 = undistort_img(img2, K, dist, stdInt)

    p1, p2, des = ORB_matching(img1, img2)

    X, des, T, uv1, uv2, E = generate_model(p1, p2, K, des)

    T, X = bundle_adjustment(T, X, uv1, uv2, K)

    save_model(X, des)

    #Plotting results
    img1 = plt.imread("../iCloud Photos/IMG_3980.JPEG")/255.
    img2 = plt.imread("../iCloud Photos/IMG_3981.JPEG")/255.

    # np.random.seed(123) # Comment out to get a random selection each time
    plot_point_cloud(X, uv1, img1)
    draw_correspondences(img1, img2, uv1, uv2, F_from_E(E, K), sample_size=8)
    visualize_query_res(X, uv2[:2,:], K, plt.imread("../iCloud Photos/IMG_3981.JPEG"), T)
    plt.show()