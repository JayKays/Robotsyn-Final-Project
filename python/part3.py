
from Monte_Carlo import *
from localize import *
from visualize_query_results import visualize_query_res



def task31(K, X, model_des, query_img):

    p, J, world_points, img_points = localize(query_img, X, model_des, K)

    visualize_query_res(world_points, img_points, K, query_img, pose(p))
    # print(pose(p))

    return

def task32(K, X, model_des, query_img):
    
    p, J, _, _ = localize(query_img, X, model_des, K)
    # print(pose(p))
    std = pose_std(J)

    return unit_convertion(std)

def task33(K, X, model_des, query_img):
    
    p, J, _, _ = localize(query_img, X, model_des, K, weighted = True)

    print(pose(p))

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

    K = np.loadtxt("../hw5_data_ext/K.txt")
    X = np.loadtxt("../3D_model/3D_points.txt")
    # X[:3,:] *= 6.2
    model_des = np.loadtxt("../3D_model/descriptors").astype("float32")
    query_img = cv.imread("../hw5_data_ext/IMG_8224.jpg")

    task31(K, X, model_des, query_img)
    # std1 = task32(K, X, model_des, query_img)
    # std2 = task33(K, X, model_des, query_img)
    # task34(K, X, model_des, query_img)

    # print(std1)
