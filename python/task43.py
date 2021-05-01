import numpy as np
from matplotlib import pyplot as plt
from cv2 import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from HW5 import *
from util import *
from visualize_query_results import *
from part2 import *

def ORB_matching(img1, img2):
    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures = 40000) #specifying maximum nr of keypoints to locate

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

    # Need to draw only good matches, using 3000 first
    p1 = np.array([kp1[m.queryIdx].pt for m in matches[:3000]])
    p2 = np.array([kp2[m.trainIdx].pt for m in matches[:3000]])

    des = des1[[m.queryIdx for m in matches[:3000]], :]

    print(f"Found {len(matches)} matches. Using {len(p1)} matches with shortest distance.")

    # draw first 3000 matches
    # img3 = cv.drawMatches(image1_gray, kp1, image2_gray, kp2, matches[:3000], image2_gray, flags = 2)
    # plt.imshow(img3),plt.show()

    return p1, p2, des

if __name__ == "__main__":

    img1 = cv.imread('../iCloud Photos/IMG_3980.JPEG')
    img2 = cv.imread('../iCloud Photos/IMG_3981.JPEG')

    K = np.loadtxt("cam_matrix.txt")

    p1, p2, des = ORB_matching(img1, img2)

    X, des, T, uv1, uv2, E = generate_model(p1, p2, K, des)

    T, X = bundle_adjustment(T, X, uv1, uv2, K)

    save_model(X, des, Path('../3D_model'))

    #Plotting results
    img1 = plt.imread("../iCloud Photos/IMG_3980.JPEG")/255.
    img2 = plt.imread("../iCloud Photos/IMG_3981.JPEG")/255.

    # np.random.seed(123) # Comment out to get a random selection each time
    plot_point_cloud(X, uv1, img1)
    draw_correspondences(img1, img2, uv1, uv2, F_from_E(E, K), sample_size=8)
    visualize_query_res(X, X, uv2[:2,:], K, img2, T)
    plt.show()