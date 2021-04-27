from part2 import *
from HW5 import *
from util import *
from localize import *



def sig_p(Jac):
    return np.linalg.inv(Jac.T @ np.eye(jac.shape[0]) @ Jac)

def pose_std(sig_p):
    return np.linalg.diagonal(sig_p**2)

def unit_convertion(pose_std):
    return pose_std

def pose(p):

    angles = p[:3]
    translate = p[3:]

    T = np.eye(4)

    R,_ = cv.Rodrigues(angles)

    T[:3,:3] = R
    T[:3,-1] = translate
    return T

def residual_image(p, K, X, uv):
    "P.shape = 6"
    T = pose(p)
    
    uv_hat = project(K, X)
    r = uv_hat - uv

    return np.ravel(r.T)


if __name__ == "__main__":

    img = cv.imread("../hw5_data_ext/IMG_8210.jpg")

    X = np.loadtxt("./part3_data/X.txt")
    K = np.loadtxt("../hw5_data_ext/K.txt")
    T = np.loadtxt("./part3_data/T.txt")
    uv = np.loadtxt("./part3_data/uv2.txt")

    # T, _ =  localize(img, X, )
    R, _ = cv.Rodrigues(T[:3,:3])
    t = T[:3,-1]

    print(R.shape)
    
    p = np.hstack((R[:,0].T,t))

    res_fun = lambda p: residual_image(p, K, X, uv[:2,:])
    jac = jacobian(res_fun, p, 1e-5)

    print(np.count_nonzero(jac))
    sigma_p = sig_p(jac)
    std = pose_std(sig_p)
