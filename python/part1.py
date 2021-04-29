import numpy as np
from cv2 import cv2 as cv
import glob
from matplotlib import pyplot as plt
import numpy as np

def calibration(images):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*10,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    # images = glob.glob('../calibration_photos2/*.JPEG') # IMG_3896 to IMG_3914

    #images = images[:10] + images[12:] # comment in remove shity pictures 11 and 12

    for fname in images:

        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,10), None)
        # If found, add object points, image points (after refining them)
        if ret == True:

            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,10), corners2, ret)
            img_resized = cv.resize(img, (1400, 700))    
            cv.imshow('img', img_resized)
            cv.waitKey(10)

    cv.destroyAllWindows()

    ret, K, dist, rvecs, tvecs, std_int, std_ext, pVE = \
        cv.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1], None, None)

    mean_error = []
    error_vecs = np.zeros((70*len(images), 2))   # vertical stack of [x, y] errors for all points in all pictures
    for i in range(len(objpoints)): # calculating errors
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error.append(error)

        imgpoints2 = np.array(imgpoints2)   # converting to numpy arrays to get shit to work cuz fk normal arrays
        imgpoints2 = imgpoints2[:,0,:]
        imgpoints1 = np.array(imgpoints[i])
        imgpoints1 = imgpoints1[:,0,:]
        error_vecs[i*70:(i+1)*70, :] = imgpoints1 - imgpoints2

    fig = plt.figure(1)       # mean reprojection error plot
    img_nr = [f"{i+1}" for i in range(len(images))]
    plt.bar(img_nr, mean_error)
    plt.ylabel("mean reprojection error")
    plt.xlabel("image number")
    plt.savefig("Calibration_errors")
    plt.show()

    fig2 = plt.figure(2)
    plt.scatter(error_vecs[:,0], error_vecs[:,1])
    plt.ylabel("y error")
    plt.xlabel("x error")
    plt.savefig("Reprojection_scatter")
    plt.show()

    # _,_,_,_,_,stdDeviationsIntrinsics,stdDeviationsExtrinsics,perViewErrors = \
    #     cv.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1], mtx, dist)

    print('Standard errors')
    print('Focal length and principal point')
    print('--------------------------------')
    print('fx: %g +/- %g' % (K[0,0], std_int[0]))
    print('fy: %g +/- %g' % (K[1,1], std_int[1]))
    print('cx: %g +/- %g' % (K[0,2], std_int[2]))
    print('cy: %g +/- %g' % (K[1,2], std_int[3]))
    print('Distortion coefficients')
    print('--------------------------------')
    print('k1: %g +/- %g' % (dist[0,0], std_int[4]))
    print('k2: %g +/- %g' % (dist[0,1], std_int[5]))
    print('p1: %g +/- %g' % (dist[0,2], std_int[6]))
    print('p2: %g +/- %g' % (dist[0,3], std_int[7]))
    print('k3: %g +/- %g' % (dist[0,4], std_int[8]))
    
    np.savetxt('cam_matrix.txt', K)
    np.savetxt('dist.txt', dist)
    np.savetxt('stdInt.txt', std_int)
    np.savetxt('stdExt.txt', std_ext)

def undistort(K, dist,stdInt):
    img = cv.imread('../calibration_photos/IMG_3896.JPEG')
    img_resized = cv.resize(img, (1400, 700))

    normal_dist = np.random.normal(dist, stdInt.T[5:10])

    cv.imshow('vanilla image', img_resized)
    cv.waitKey(10000)
    cv.destroyAllWindows()

    h, w = img.shape[:2]

    #Undistort
    undistorted = cv.undistort(img, K, dist, None, newK)

    dist[4:9] = dist[4:9] + stdInt[4:9].T

    # undistort
    dst = cv.undistort(img, K, dist, None, newcameramtx)
    

def undistort_img(img, K, distortion, dist_std, random_dist = False):
    '''Undistorts and displays a given image'''
    
    if random_dist:
        dist = np.random.normal(distortion, dist_std)
    else:
        dist = distortion

    h, w = img.shape[:2]
    newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

    #Undistort
    undistorted = cv.undistort(img, K, dist, None, newK)

    #Crop
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]
    # undist_resized = cv.resize(undistorted, (1440, 960))

    #Display undistorted image
    # cv.imshow('undistorted', undist_resized)
    # cv.waitKey(-1)

    # cv.destroyAllWindows()

    return undistorted


if __name__ == "__main__":
    images = glob.glob('../iCloud Photos/Calib23/*.JPEG')
    calibration(images)

    K = np.loadtxt('cam_matrix.txt')
    dist = np.loadtxt('dist.txt')
    stdInt = np.loadtxt('stdInt.txt')
    img = cv.imread('../iCloud Photos/Calib23/IMG_3991.JPEG')

    # print(stdInt)
    # print(dist)

    # undistort(K, dist, stdInt)
    # img_resized = cv.resize(img, (1440, 960))
    # cv.imshow('Original image', img_resized)
    # cv.waitKey(-1)

    # for i in range(10):
    #     undistort_img(img, K, dist[:], stdInt.T[5:10], random_dist=True)
