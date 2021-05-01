import numpy as np
import matplotlib.pyplot as plt

def jacobian(f,x,eps):
    '''Calculates central difference estimation
    of jacobian for given function f(x) '''

    gradient = lambda h: (f(x + h) - f(x-h))/(2*eps)
    
    return np.apply_along_axis(gradient, 0, np.eye(x.shape[0])*eps)

def project(K, X):
    uvw = K@X[:3,:]
    uvw /= uvw[2,:]
    return uvw[:2,:]

def project_camera_frame(K, T, scale):
    """
    Draw the axes of T and a pyramid, representing the camera.
    """
    s = scale
    X = []
    X.append(np.array([0,0,0,1]))
    X.append(np.array([-s,-s,1.5*s,1]))
    X.append(np.array([+s,-s,1.5*s,1]))
    X.append(np.array([+s,+s,1.5*s,1]))
    X.append(np.array([-s,+s,1.5*s,1]))
    X.append(np.array([5.0*s,0,0,1]))
    X.append(np.array([0,5.0*s,0,1]))
    X.append(np.array([0,0,5.0*s,1]))
    X = np.array(X).T
    u,v = project(K, T@X)
    lines = [(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1)]
    plt.plot([u[0], u[5]], [v[0], v[5]], color='#ff5555', linewidth=2)
    plt.plot([u[0], u[6]], [v[0], v[6]], color='#33cc55', linewidth=2)
    plt.plot([u[0], u[7]], [v[0], v[7]], color='#44aaff', linewidth=2)
    for (i,j) in lines:
        plt.plot([u[i], u[j]], [v[i], v[j]], color='black')

def draw_model_and_query_pose(X, T_m2q, K,
    lookat=np.array((0.0, 0.0, 0.0)),
    lookfrom=np.array((0.0, 0.0, -15.0)),
    point_size=10,
    frame_size=0.5,
    c=None):

    """
              X: Point cloud model of [shape (3 or 4)xN].
          T_m2q: Transformation from model to query camera coordinates (e.g. as obtained from OpenCV's solvePnP).
              K: Intrinsic matrix for the virtual 'figure camera'.
    lookat|from: The viewing target and origin of the virtual figure camera.
     point_size: Radius of a point (in pixels) that is 1 unit away. (Points further away will appear smaller.)
     frame_size: The length (in model units) of the camera and coordinate frame axes.
              c: Color associated with each point in X [shape Nx3].
    """
    if c is None:
        uv = project(K, X)
        I = plt.imread('../iCloud Photos/IMG_3980.JPEG')/255.
        c = I[uv[1,:].astype(np.int32), uv[0,:].astype(np.int32), :]

    assert X.ndim == 2, 'X must be a (3 or 4)xN array'
    assert X.shape[1] > 0, 'X must have at least one point'

    X = X.copy()
    if X.shape[0] == 3:
        X = np.vstack([X, np.ones_like(X[0,:])])
    else:
        X = X/X[3,:]

    if c is None:
        c = X[1,:]
    else:
        c = c.copy()
        if np.max(c) > 1.0:
            c = c / 256.0

    # Create transformation from model to 'figure camera'
    T_f2m = np.eye(4)
    T_f2m[:3,2] = (lookat - lookfrom)
    T_f2m[:3,2] /= np.linalg.norm(T_f2m[:3,2])
    T_f2m[:3,0] = np.cross(np.array((0,1,0)), T_f2m[:3,2])
    T_f2m[:3,0] /= np.linalg.norm(T_f2m[:3,0])
    T_f2m[:3,1] = np.cross(T_f2m[:3,2], T_f2m[:3,0])
    T_f2m[:3,1] /= np.linalg.norm(T_f2m[:3,1])
    T_f2m[:3,3] = lookfrom
    T_m2f = np.linalg.inv(T_f2m)

    # Transform point cloud model into 'figure camera' frame
    X_f = T_m2f@X
    visible = X_f[2,:] > 0
    not_visible = X_f[2,:] < 0
    X_f = X_f[:,visible]
    if np.sum(not_visible) > np.sum(visible):
        print('[draw_model_and_query_pose] Most of point cloud is behind camera, is that intentional?')

    # Project point cloud with depth sorting
    T_q2m = np.linalg.inv(T_m2q)
    u = K@X_f[:3,:]
    u = u[:2,:]/u[2,:]
    i = np.argsort(-X_f[2,:])
    plt.scatter(*u[:,i], c=c[i], s=(point_size**2)/X_f[2,:], rasterized=True)

    # Project the coordinate frame axes of the localized query camera
    project_camera_frame(K, T_m2f@T_q2m, scale=frame_size)

    plt.axis('image')

    # Arbitrarily set axis limits to 2*principal point
    plt.xlim([0.0, K[0,2]*2])
    plt.ylim([K[1,2]*2, 0.0])
