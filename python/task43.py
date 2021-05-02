import numpy as np
from matplotlib import pyplot as plt
from cv2 import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from HW5 import *
from util import *
from visualize_query_results import *
from part2 import *
from localize import *

def ORB_matching(img1, img2):
    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures = 120000) #specifying maximum nr of keypoints to locate

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

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Need to draw only good matches, using 4500 best
    p1 = np.array([kp1[m.queryIdx].pt for m in matches[:4500]])
    p2 = np.array([kp2[m.trainIdx].pt for m in matches[:4500]])

    des = des1[[m.queryIdx for m in matches[:4500]], :]

    print(f"Found {len(matches)} matches. Using {len(p1)} matches with shortest distance.")

    return p1, p2, des



def match_image_to_model(X, model_des, query_img, threshold = 0.75):


    orb = cv.ORB_create(nfeatures = 120000) #specifying maximum nr of keypoints to locate

    img_gray = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)

    # find the keypoints and descriptors with ORB
    query_kp, query_des = orb.detectAndCompute(img_gray, None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(model_des, query_des)

    # Need to draw only good matches
    matches = sorted(matches, key = lambda x:x.distance)
    
    matched_2D_points = np.array([query_kp[m.trainIdx].pt for m in matches[:4000]])
    matched_3D_points = X[:,[m.queryIdx for m in matches[:4000]]]

    return matched_2D_points, matched_3D_points

if __name__ == "__main__":
    np.random.seed(0) # Comment out to get a random selection each time

    img1 = cv.imread('../iCloud Photos/IMG_3980.JPEG')
    img2 = cv.imread('../iCloud Photos/IMG_3981.JPEG')
    img3 = cv.imread('../iCloud Photos/IMG_3982.JPEG')

    K = np.loadtxt("cam_matrix.txt")

    p1, p2, des = ORB_matching(img1, img2)

    X, des, T, uv1, uv2, E = generate_model(p1, p2, K, des)

    T, X = bundle_adjustment(T, X, uv1, uv2, K)

    img_points, world_points = match_image_to_model(X, des, img3)
    p, world_points, img_points, J, R0 = estimate_pose(img_points.T, world_points, K)
    
    X[:3,:] *= 8

    save_model(X, des, Path('../ORB_model'))

    #Plotting results
    img1 = plt.imread("../iCloud Photos/IMG_3980.JPEG")/255.
    img2 = plt.imread("../iCloud Photos/IMG_3981.JPEG")/255.

    visualize_query_res(X, world_points, img_points, K, img3, pose(p,R0))
    plt.savefig("ORB_kuk.eps", format='eps')
    plt.show()
