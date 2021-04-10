import numpy as np
from matplotlib import pyplot as plt
from cv2 import cv2 as cv
from HW5 import *




def SIFT_matching(img1,img2, threshold = 0.75):

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
            good.append([m])
    print(good[0])
    print(f"Found {len(good)} matches with distance thrshold = {threshold}")
    # cv.drawMatchesKnn expects list of lists as matches.

    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = cv.DrawMatchesFlags_DEFAULT)

    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3), plt.show()

    return good




if __name__ == "__main__":

    img1 = cv.imread("../hw5_data_ext/IMG_8207.jpg")
    img2 = cv.imread("../hw5_data_ext/IMG_8228.jpg")
    K = np.loadtxt("../hw5_data_ext/K.txt")
    
    matches = SIFT_matching(img1, img2)
