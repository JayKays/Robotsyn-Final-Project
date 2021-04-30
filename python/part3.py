
from Monte_Carlo import *
from localize import *
from visualize_query_results import visualize_query_res
from part1 import undistort_img


def task31(K, X, model_des, query_img):

    p, J, world_points, img_points = localize(query_img, X, model_des, K)

    visualize_query_res(X, world_points, img_points, K, query_img, pose(p))
    # print(pose(p))

    return

def task32(K, X, model_des, query_img):
    
    p, J, _, _ = localize(query_img, X, model_des, K)
    # print(pose(p))
    std = pose_std(J)

    return unit_convertion(std)

def task33(K, X, model_des, query_img):
    
    p, J, _, _ = localize(query_img, X, model_des, K, weighted = True)

    # print(pose(p))

    std = pose_std(J)

    return unit_convertion(std)


def task34(K, X, model_des, query_img):

    p0, _, X, uv = localize(query_img, X, model_des, K)

    std1 = monte_carlo_std(K, [50, 0.1, 0.1], p0, uv, X)
    std2 = monte_carlo_std(K, [0.1, 50, 0.1], p0, uv, X)
    std3 = monte_carlo_std(K, [0.1, 0.1, 50], p0, uv, X)

    print(std1)
    print(std2)
    print(std3)

    return

if __name__ == "__main__":
    np.random.seed(0)

    HW5_model = False
    
    if HW5_model:
        K = np.loadtxt("../hw5_data_ext/K.txt")
        X = np.loadtxt("../HW5_3D_model/3D_points.txt")
        model_des = np.loadtxt("../HW5_3D_model/descriptors").astype("float32")
        query_img = cv.imread("../hw5_data_ext/IMG_8211.jpg")

    else:
        K = np.loadtxt("cam_matrix.txt")
        X = np.loadtxt("../3D_model/3D_points.txt")
        model_des = np.loadtxt("../3D_model/descriptors").astype("float32")
        distortion = np.loadtxt('dist.txt')
        query_img = cv.imread('../iCloud Photos/IMG_3982.JPEG')
        # dist_std = np.loadtxt('stdInt.txt')

    # undistort_img(img, K, distortion, None)
    # img1 = undistort_img(cv.imread('../iCloud Photos/IMG_4001.JPEG'), K ,distortion, None)
    # img2 = undistort_img(cv.imread('../iCloud Photos/IMG_3981.JPEG'), K ,distortion, None)
    # img3 = undistort_img(cv.imread('../iCloud Photos/IMG_4003.JPEG'), K ,distortion, None)
    # img1 = cv.imread('../iCloud Photos/IMG_3980.JPEG')
    # img2 = cv.imread('../iCloud Photos/IMG_3981.JPEG')
    # img3 = cv.imread('../iCloud Photos/IMG_3982.JPEG')

    # img1 = undistort_img(img1, K, distortion, None)
    # img2 = undistort_img(img2, K, distortion, None)
    # img3 = undistort_img(img3, K, distortion, None)


    # task31(K, X, model_des, img1)
    # task31(K, X, model_des, img2)
    # task31(K, X, model_des, img3)

    std1 = task32(K, X, model_des, query_img)
    std2 = task33(K, X, model_des, query_img)
    # task34(K, X, model_des, img3)

    print(std1)
    print(std2)