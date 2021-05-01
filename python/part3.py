from Monte_Carlo import monte_carlo_std
from localize import *
from visualize_query_results import visualize_query_res
from part1 import undistort_img


def task31(K, X, model_des, query_img, using_rootsift):

    p, J, world_points, img_points, R0 = localize(query_img, X, model_des, K, using_rootsift)

    T = pose(p, R0)

    visualize_query_res(X, world_points, img_points, K, query_img, T)
    # print(p)

    return

def task32(K, X, model_des, query_img, using_rootsift):
    
    _, J, _, _, _ = localize(query_img, X, model_des, K, using_rootsift)
    # print(pose(p))
    std = pose_std(J)

    return unit_convertion(std)

def task33(K, X, model_des, query_img, using_rootsift):
    
    _, J, _, _, _ = localize(query_img, X, model_des, K, using_rootsift, weighted = True)

    # print(pose(p))

    std = pose_std(J)

    return unit_convertion(std)


def task34(K, X, model_des, query_img, using_rootsift):

    p0, _, X, uv, R0 = localize(query_img, X, model_des, K, using_rootsift)

    std1 = monte_carlo_std(K, [50, 0.1, 0.1], p0, uv, X, R0)
    std2 = monte_carlo_std(K, [0.1, 50, 0.1], p0, uv, X, R0)
    std3 = monte_carlo_std(K, [0.1, 0.1, 50], p0, uv, X, R0)

    print(std1)
    print(std2)
    print(std3)

    return

def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

if __name__ == "__main__":
    np.random.seed(0)

    HW5_model = False
    using_rootsift = False
    
    if HW5_model:
        K = np.loadtxt("../hw5_data_ext/K.txt")
        X = np.loadtxt("../HW5_3D_model/3D_points.txt")
        model_des = np.loadtxt("../HW5_3D_model/descriptors").astype("float32")
        query_img = cv.imread("../hw5_data_ext/IMG_8214.jpg")
    else:
        K = np.loadtxt("cam_matrix.txt")
        X = np.loadtxt("../3D_model/3D_points.txt")
        model_des = np.loadtxt("../3D_model/descriptors").astype("float32")
        distortion = np.loadtxt('dist.txt')
        query_img = cv.imread('../iCloud Photos/IMG_3982.JPEG')
        # dist_std = np.loadtxt('stdInt.txt')

    # undistort_img(img, K, distortion, None)
    img1 = cv.imread('../iCloud Photos/IMG_3982.JPEG')
    img2 = cv.imread('../iCloud Photos/IMG_3983.JPEG')
    img3 = cv.imread('../iCloud Photos/IMG_4003.JPEG')

    task31(K, X, model_des, img1, using_rootsift)
    task31(K, X, model_des, img2, using_rootsift)
    task31(K, X, model_des, img3, using_rootsift)

    # std1 = task32(K, X, model_des, img1, using_rootsift)
    # std2 = task32(K, X, model_des, img2, using_rootsift)
    # std3 = task32(K, X, model_des, img3, using_rootsift)

    # std1 = task33(K, X, model_des, img1, using_rootsift)
    # std2 = task33(K, X, model_des, img2, using_rootsift)
    # std3 = task33(K, X, model_des, img3, using_rootsift)

    # task34(K, X, model_des, img3)

    # print(std1)
    # print(std2)
    # print(std3)
