import numpy as np
from matplotlib import pyplot as plt

"Contains all usefull functions from HW5"

def project(K, X):
    """
    Computes the pinhole projection of a (3 or 4)xN array X using
    the camera intrinsic matrix K. Returns the pixel coordinates
    as an array of size 2xN.
    """
    uvw = K@X[:3,:]
    uvw /= uvw[2,:]
    return uvw[:2,:]

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


def estimate_E(xy1, xy2):
    n = xy1.shape[1]
    A = np.empty((n, 9))
    for i in range(n):
        x1,y1 = xy1[:2,i]
        x2,y2 = xy2[:2,i]
        A[i,:] = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]

    _,_,VT = np.linalg.svd(A)
    return np.reshape(VT[-1,:], (3,3))

def get_num_ransac_trials(sample_size = 8, confidence = 0.99, inlier_fraction = 0.5):
    return int(np.log(1 - confidence)/np.log(1 - inlier_fraction**sample_size))

def estimate_E_ransac(xy1, xy2, K, distance_threshold = 4, num_trials = get_num_ransac_trials()):

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

def draw_point_cloud(X, I1, uv1, xlim, ylim, zlim, name = '3D point cloud'):
    assert uv1.shape[1] == X.shape[1], 'If you get this error message in Task 4, it probably means that you did not extract the inliers of all the arrays (uv1,uv2,xy1,xy2) before calling draw_point_cloud.'

    # We take I1 and uv1 as arguments in order to assign a color to each
    # 3D point, based on its pixel coordinates in one of the images.
    c = I1[uv1[1,:].astype(np.int32), uv1[0,:].astype(np.int32), :]

    # Matplotlib doesn't let you easily change the up-axis to match the
    # convention we use in the course (it assumes Z is upward). So this
    # code does a silly rearrangement of the Y and Z arguments.
    plt.figure(name, figsize=(6,6))
    ax = plt.axes(projection='3d')
    ax.scatter(X[0,:], X[2,:], X[1,:], c=c, marker='.', depthshade=False)
    ax.grid(False)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim([ylim[1], ylim[0]])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    plt.title('[Click, hold and drag with the mouse to rotate the view]')

def hline(l, **args):
    """
    Draws a homogeneous 2D line.
    You must explicitly set the figure xlim, ylim before or after using this.
    """

    lim = np.array([-1e8, +1e8]) # Surely you don't have a figure bigger than this!
    a,b,c = l
    if np.absolute(a) > np.absolute(b):
        x,y = -(c + b*lim)/a, lim
    else:
        x,y = lim, -(c + a*lim)/b
    plt.plot(x, y, **args)

def draw_correspondences(I1, I2, uv1, uv2, F, sample_size=8):
    """
    Draws a random subset of point correspondences and their epipolar lines.
    """

    assert uv1.shape[0] == 3 and uv2.shape[0] == 3, 'uv1 and uv2 must be 3 x n arrays of homogeneous 2D coordinates.'
    sample = np.random.choice(range(uv1.shape[1]), size=sample_size, replace=False)
    uv1 = uv1[:,sample]
    uv2 = uv2[:,sample]
    n = uv1.shape[1]
    uv1 /= uv1[2,:]
    uv2 /= uv2[2,:]

    l1 = F.T@uv2
    l2 = F@uv1

    colors = plt.cm.get_cmap('Set2', n).colors
    plt.figure('Correspondences', figsize=(10,4))
    plt.subplot(121)
    plt.imshow(I1)
    plt.xlabel('Image 1')
    plt.scatter(*uv1[:2,:], s=100, marker='x', c=colors)
    for i in range(n):
        hline(l1[:,i], linewidth=1, color=colors[i], linestyle='--')
    plt.xlim([0, I1.shape[1]])
    plt.ylim([I1.shape[0], 0])

    plt.subplot(122)
    plt.imshow(I2)
    plt.xlabel('Image 2')
    plt.scatter(*uv2[:2,:], s=100, marker='o', zorder=10, facecolor='none', edgecolors=colors, linewidths=2)
    for i in range(n):
        hline(l2[:,i], linewidth=1, color=colors[i], linestyle='--')
    plt.xlim([0, I2.shape[1]])
    plt.ylim([I2.shape[0], 0])
    plt.tight_layout()
    plt.suptitle('Point correspondences and associated epipolar lines (showing %d randomly drawn pairs)' % sample_size)