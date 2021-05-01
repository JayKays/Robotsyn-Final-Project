import matplotlib.pyplot as plt
import numpy as np
from util import *
from localize import *

def visualize_query_res(X, X_inliers, u, K, I, T_m2q, uv = None, model_img = None):

    assert X.shape[0] == 4
    assert u.shape[0] == 2

    # for setting the color of the points clouds
    if uv is not None and model_img is not None:
        c = model_img[uv[1,:].astype(np.int32), uv[0,:].astype(np.int32), :]
    else: 
        c = None

    # These control the location and the viewing target
    # of the virtual figure camera, in the two views.
    # You will probably need to change these to work
    # with your scene.
    
    lookfrom1 = np.array((0,-20,-5))
    lookat1   = np.array((0,0,6))
    lookfrom2 = np.array((25,-15,-10))
    lookat2   = np.array((0,0,10))

    u_inliers = u
    # X_inliers = X

    u_hat = project(K, T_m2q @ X_inliers)
    e = np.linalg.norm(u_hat - u_inliers, axis=0)

    fig = plt.figure(figsize=(10,8))

    plt.subplot(221)
    plt.imshow(I)
    plt.scatter(*u_hat, marker='+', c=e)
    plt.xlim([0, I.shape[1]])
    plt.ylim([I.shape[0], 0])
    plt.colorbar(label='Reprojection error (pixels)')
    plt.title('Query image and reprojected points')

    plt.subplot(222)
    plt.hist(e, bins=50)
    plt.xlabel('Reprojection error (pixels)')

    plt.subplot(223)
    draw_model_and_query_pose(X, T_m2q, K, lookat1, lookfrom1, c=c)
    plt.title('Model and localized pose (top view)')

    plt.subplot(224)
    draw_model_and_query_pose(X, T_m2q, K, lookat2, lookfrom2, c=c)
    plt.title('Model and localized pose (side view)')

    plt.tight_layout()
    plt.show()


