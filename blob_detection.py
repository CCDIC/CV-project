
import os
import numpy as np
import scipy.ndimage
from common import (
    read_img, save_img, find_maxima,
    visualize_maxima, visualize_scale_space
)

def gaussian_filter(image, sigma):
    """
    对图像应用二维高斯滤波器。
    """
    return scipy.ndimage.gaussian_filter(image, sigma=sigma, mode='reflect')

def detect_cells(image, sigma1, sigma2, k_xy=10):
    """
    使用 DoG 方法检测图像中的细胞（blob）。
    """
    gauss1 = gaussian_filter(image, sigma=sigma1)
    gauss2 = gaussian_filter(image, sigma=sigma2)
    DoG = gauss2 - gauss1
    maxima = find_maxima(DoG, k_xy=k_xy)
    return DoG, maxima

def main():
    # ---------- ✅ Task 3 & 4 ----------
    image = read_img('polka.png')
    os.makedirs("./log_filter", exist_ok=True)

    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([
        [0, 0, 3, 2, 2, 2, 3, 0, 0],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [2, 5, 0, -23, -40, -23, 0, 5, 2],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [0, 0, 3, 2, 2, 2, 3, 0, 0]
    ])
    filtered_LoG1 = scipy.ndimage.convolve(image, kernel_LoG1, mode='reflect')
    filtered_LoG2 = scipy.ndimage.convolve(image, kernel_LoG2, mode='reflect')

    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    sigma = 50
    k = 53 / 50
    gauss_1 = gaussian_filter(image, sigma=sigma)
    gauss_2 = gaussian_filter(image, sigma=sigma * k)
    DoG = gauss_2 - gauss_1
    os.makedirs('./log_filter/dog', exist_ok=True)
    save_img(DoG, './log_filter/dog/q1_dog.png')

    data = np.load("log1d.npz")
    log50 = data['log50']
    gauss50 = data['gauss50']
    gauss53 = data['gauss53']
    dog_1d = gauss53 - gauss50
    diff = np.abs(log50 - dog_1d)
    print(f"最大差异: {np.max(diff):.6f}")
    print(f"平均差异: {np.mean(diff):.6f}")

    os.makedirs('./polka_detections', exist_ok=True)
    sigma_1_small, sigma_2_small = 2.8,3.0
    DoG_small = gaussian_filter(image, sigma_2_small) - gaussian_filter(image, sigma_1_small)
    maxima_small = find_maxima(DoG_small, k_xy=13)
    visualize_scale_space(DoG_small, sigma_1_small, sigma_2_small / sigma_1_small,
                          './polka_detections/polka_small_DoG.png')
    visualize_maxima(image, maxima_small, sigma_1_small, sigma_2_small / sigma_1_small,
                     './polka_detections/polka_small.png')

    sigma_1_large, sigma_2_large =8,9
    DoG_large = gaussian_filter(image, sigma_2_large) - gaussian_filter(image, sigma_1_large)
    maxima_large = find_maxima(DoG_large, k_xy=10)
    visualize_scale_space(DoG_large, sigma_1_large, sigma_2_large / sigma_1_large,
                          './polka_detections/polka_large_DoG.png')
    visualize_maxima(image, maxima_large, sigma_1_large, sigma_2_large / sigma_1_large,
                     './polka_detections/polka_large.png')

    print("✅ 单尺度斑点检测完成！")

    # ---------- ✅ Task 5: 细胞检测 ----------
    print("🔬 开始执行 Task 5：细胞数量检测")
    os.makedirs('./cell_detections', exist_ok=True)
    cell_dir = './cells'
    cell_imgs = sorted([f for f in os.listdir(cell_dir) if f.endswith('.png')])[:4]

    for fname in cell_imgs:
        path = os.path.join(cell_dir, fname)
        img = read_img(path)
        sigma1, sigma2 = 8, 8.1  # 可微调
        DoG_cell, maxima_cell = detect_cells(img, sigma1, sigma2, k_xy=7)
        output_path = os.path.join('./cell_detections', f'detect_{fname}')
        visualize_maxima(img, maxima_cell, sigma1, sigma2 / sigma1, output_path)
        print(f"{fname} 检测到细胞数量: {len(maxima_cell)}")

if __name__ == '__main__':
    main()
