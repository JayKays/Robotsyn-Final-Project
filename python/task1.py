import numpy as np
from cv2 import cv2 as cv
import glob
from matplotlib import pyplot as plt
import numpy as np
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*10,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('../calibration_photos/*.JPEG')

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

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
mean_error = []
print(mtx)
error_vecs = np.zeros((70*19, 2))   # vertical stack of [x, y] errors for all points in all pictures
for i in range(len(objpoints)): # calculating errors
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error.append(error)

    imgpoints2 = np.array(imgpoints2)   # converting to numpy arrays to get shit to work cuz fk normal arrays
    imgpoints2 = imgpoints2[:,0,:]
    imgpoints1 = np.array(imgpoints[i])
    imgpoints1 = imgpoints1[:,0,:]
    error_vecs[i*70:(i+1)*70, :] = imgpoints1 - imgpoints2

fig = plt.figure(1)       # mean reprojection error plot
img_nr = [f"{i+1}" for i in range(19)]
plt.bar(img_nr, mean_error)
plt.ylabel("mean reprojection error")
plt.xlabel("image number")
plt.show()

fig2 = plt.figure(2)
plt.scatter(error_vecs[:,0], error_vecs[:,1])
plt.show()