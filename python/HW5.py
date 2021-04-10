import numpy as np
from matplotlib import pyplot as plt

def SE3(R,t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def decompose_E(E):
    """
    Computes the four possible decompositions of E into a relative
    pose, as described in Szeliski 7.2.

    Returns a list of 4x4 transformation matrices.
    """
    U,_,VT = np.linalg.svd(E)
    R90 = np.array([[0, -1, 0], [+1, 0, 0], [0, 0, 1]])
    R1 = U @ R90 @ VT
    R2 = U @ R90.T @ VT
    if np.linalg.det(R1) < 0: R1 = -R1
    if np.linalg.det(R2) < 0: R2 = -R2
    t1, t2 = U[:,2], -U[:,2]
    return [SE3(R1,t1), SE3(R1,t2), SE3(R2, t1), SE3(R2, t2)]

def get_num_ransac_trials(sample_size, confidence, inlier_fraction):
    return int(np.log(1 - confidence)/np.log(1 - inlier_fraction**sample_size))

def estimate_E(xy1, xy2):
    n = xy1.shape[1]
    A = np.empty((n, 9))
    for i in range(n):
        x1,y1 = xy1[:2,i]
        x2,y2 = xy2[:2,i]
        A[i,:] = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]

    _,_,VT = np.linalg.svd(A)
    return np.reshape(VT[-1,:], (3,3))

def estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials):

    # Tip: The following snippet extracts a random subset of 8
    # correspondences (w/o replacement) and estimates E using them.
    #   sample = np.random.choice(xy1.shape[1], size=8, replace=False)
    #   E = estimate_E(xy1[:,sample], xy2[:,sample])

    uv1 = K@xy1
    uv2 = K@xy2

    print('Running RANSAC with %g inlier threshold and %d trials...' % (distance_threshold, num_trials), end='')
    best_num_inliers = -1
    for i in range(num_trials):
        sample = np.random.choice(xy1.shape[1], size=8, replace=False)
        E_i = estimate_E(xy1[:,sample], xy2[:,sample])
        d_i = epipolar_distance(F_from_E(E_i, K), uv1, uv2)
        inliers_i = np.absolute(d_i) < distance_threshold
        num_inliers_i = np.sum(inliers_i)
        if num_inliers_i > best_num_inliers:
            best_num_inliers = num_inliers_i
            E = E_i
            inliers = inliers_i
    print('Done!')
    print('Found solution with %d/%d inliers' % (np.sum(inliers), xy1.shape[1]))
    return E, inliers

def epipolar_distance(F, uv1, uv2):
    """
    F should be the fundamental matrix (use F_from_E)
    uv1, uv2 should be 3 x n homogeneous pixel coordinates
    """
    n = uv1.shape[1]
    l2 = F@uv1
    l1 = F.T@uv2
    e = np.sum(uv2*l2, axis=0)
    norm1 = np.linalg.norm(l1[:2,:], axis=0)
    norm2 = np.linalg.norm(l2[:2,:], axis=0)
    return 0.5*e*(1/norm1 + 1/norm2)

def F_from_E(E, K):
    K_inv = np.linalg.inv(K)
    F = K_inv.T@E@K_inv
    return F

def triangulate_many(xy1, xy2, P1, P2):
    """
    Arguments
        xy: Calibrated image coordinates in image 1 and 2
            [shape 3 x n]
        P:  Projection matrix for image 1 and 2
            [shape 3 x 4]
    Returns
        X:  Dehomogenized 3D points in world frame
            [shape 4 x n]
    """
    n = xy1.shape[1]
    X = np.empty((4,n))
    for i in range(n):
        A = np.empty((4,4))
        A[0,:] = P1[0,:] - xy1[0,i]*P1[2,:]
        A[1,:] = P1[1,:] - xy1[1,i]*P1[2,:]
        A[2,:] = P2[0,:] - xy2[0,i]*P2[2,:]
        A[3,:] = P2[1,:] - xy2[1,i]*P2[2,:]
        U,s,VT = np.linalg.svd(A)
        X[:,i] = VT[3,:]/VT[3,3]
    return X

