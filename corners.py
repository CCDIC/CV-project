import os
import numpy as np
import scipy.ndimage
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# ---------------- 图像工具函数 ----------------

def read_img(path):
    """
    读取图像为灰度图像，并转换为 float32 格式的 numpy 数组。
    """
    img = Image.open(path).convert('L')  # 转为灰度图
    return np.array(img).astype(np.float32)

def save_img(img, path):
    """
    将图像归一化到 0~255 范围并保存为 uint8 PNG 文件。
    """
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"Saved: {path}")

# ---------------- ✅ 任务一：corner_score ----------------

def corner_score(image, u=0, v=0, window_size=(5, 5)):
    """
    计算图像中每个像素点的角点响应得分 E(u,v)
    E(u,v) = sum over W of (I(x,y) - I(x+u,y+v))^2
    """
    shifted = np.roll(image, shift=(v, u), axis=(0, 1))
    diff_squared = (image - shifted) ** 2
    window = np.ones(window_size)
    score = scipy.ndimage.convolve(diff_squared, window, mode='constant')
    return score

# ---------------- ✅ 任务二：harris_detector ----------------

def harris_detector(image, window_size=(5, 5), k=0.05):
    """
    实现 Harris 角点响应函数：R = det(M) - k * (trace(M))^2
    """
    Ix = scipy.ndimage.sobel(image, axis=1, mode='constant')  # dx
    Iy = scipy.ndimage.sobel(image, axis=0, mode='constant')  # dy
    Ixx, Iyy, Ixy = Ix**2, Iy**2, Ix*Iy

    window = np.ones(window_size)
    Sxx = scipy.ndimage.convolve(Ixx, window, mode='constant')
    Syy = scipy.ndimage.convolve(Iyy, window, mode='constant')
    Sxy = scipy.ndimage.convolve(Ixy, window, mode='constant')

    det = Sxx * Syy - Sxy ** 2
    trace = Sxx + Syy
    R = det - k * (trace ** 2)

    return R

# ---------------- ✅ 主函数 ----------------

def main():
    img = read_img('./grace_hopper.png')
    os.makedirs('./feature_detection', exist_ok=True)

    # ---------- 任务一：corner_score ----------
    offsets = [(0, 5), (0, -5), (5, 0), (-5, 0)]
    for u, v in offsets:
        score = corner_score(img, u=u, v=v, window_size=(5, 5))
        save_img(score, f'./feature_detection/corner_score_u{u}_v{v}.png')

    # ---------- 任务二：harris_detector ----------
    harris_response = harris_detector(img, window_size=(5, 5), k=0.05)
    save_img(harris_response, './feature_detection/harris_response.png')

    # ---------- 🔥 改进热力图 ----------
    response_clipped = np.clip(harris_response, 0, np.percentile(harris_response, 99))

    plt.figure(figsize=(8, 6))
    plt.imshow(response_clipped, cmap='hot',
               norm=colors.Normalize(vmin=0, vmax=response_clipped.max()))
    plt.colorbar(label='Harris Response')
    plt.title('Harris Corner Heatmap')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./feature_detection/harris_heatmap.png', dpi=300)
    plt.close()

    print("✅ 美化后的热力图已保存为 ./feature_detection/harris_heatmap.png")

if __name__ == "__main__":
    main()

# import os
# import numpy as np
# import scipy.ndimage
# from PIL import Image
# import matplotlib.pyplot as plt  # 用于绘制热力图
#
# # ---------------- 图像工具函数 ----------------
#
# def read_img(path):
#     """
#     读取图像为灰度图像，并转换为 float32 格式的 numpy 数组。
#     """
#     img = Image.open(path).convert('L')  # 转为灰度图
#     return np.array(img).astype(np.float32)
#
# def save_img(img, path):
#     """
#     将图像归一化到 0~255 范围并保存为 uint8 PNG 文件。
#     """
#     img = img - img.min()
#     img = img / img.max()
#     img = (img * 255).astype(np.uint8)
#     Image.fromarray(img).save(path)
#     print(f"Saved: {path}")
#
# # ---------------- ✅ 任务一：corner_score ----------------
#
# def corner_score(image, u=0, v=0, window_size=(5, 5)):
#     """
#     计算图像中每个像素点的角点响应得分 E(u,v)
#     E(u,v) = sum over W of (I(x,y) - I(x+u,y+v))^2
#
#     参数:
#     - image: 输入图像（灰度图）
#     - u, v: 偏移量
#     - window_size: 卷积窗口大小
#
#     返回:
#     - score: 响应图（H x W）
#     """
#     # 平移图像
#     shifted = np.roll(image, shift=(v, u), axis=(0, 1))
#
#     # 差值平方
#     diff_squared = (image - shifted) ** 2
#
#     # 局部窗口（全1核）
#     window = np.ones(window_size)
#
#     # 使用卷积做局部求和
#     score = scipy.ndimage.convolve(diff_squared, window, mode='constant')
#     return score
#
# # ---------------- ✅ 任务二：harris_detector ----------------
#
# def harris_detector(image, window_size=(5, 5), k=0.05):
#     """
#     实现 Harris 角点响应函数：
#     R = det(M) - k * (trace(M))^2
#
#     参数：
#     - image: 输入灰度图像 (H x W)
#     - window_size: 窗口大小
#     - k: Harris 参数（一般取 0.04~0.06）
#
#     返回：
#     - response: 每个像素的 Harris 响应值图
#     """
#     # 图像梯度（使用 Sobel）
#     Ix = scipy.ndimage.sobel(image, axis=1, mode='constant')  # dx
#     Iy = scipy.ndimage.sobel(image, axis=0, mode='constant')  # dy
#
#     # 构造结构张量分量
#     Ixx = Ix ** 2
#     Iyy = Iy ** 2
#     Ixy = Ix * Iy
#
#     # 对每个结构张量元素进行窗口加权求和（平滑）
#     window = np.ones(window_size)
#     Sxx = scipy.ndimage.convolve(Ixx, window, mode='constant')
#     Syy = scipy.ndimage.convolve(Iyy, window, mode='constant')
#     Sxy = scipy.ndimage.convolve(Ixy, window, mode='constant')
#
#     # 计算 Harris 响应值
#     det = Sxx * Syy - Sxy ** 2
#     trace = Sxx + Syy
#     R = det - k * (trace ** 2)
#
#     return R
#
# # ---------------- ✅ 主函数 ----------------
#
# def main():
#     # 读取图像
#     img = read_img('./grace_hopper.png')
#
#     # 创建输出目录
#     if not os.path.exists('./feature_detection'):
#         os.makedirs('./feature_detection')
#
#     # ---------- 任务一：corner_score ----------
#     offsets = [(0, 5), (0, -5), (5, 0), (-5, 0)]
#     window_size = (5, 5)
#     for u, v in offsets:
#         score = corner_score(img, u=u, v=v, window_size=window_size)
#         filename = f'./feature_detection/corner_score_u{u}_v{v}.png'
#         save_img(score, filename)
#
#     # ---------- 任务二：harris_detector ----------
#     harris_response = harris_detector(img, window_size=(5, 5), k=0.05)
#
#     # 保存归一化响应图（灰度图）
#     save_img(harris_response, './feature_detection/harris_response.png')
#
#     # 保存热力图（彩色）
#     plt.figure(figsize=(8, 6))
#     plt.imshow(harris_response, cmap='jet')
#     plt.colorbar(label='Harris Response')
#     plt.title('Harris Corner Response Heatmap')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig('./feature_detection/harris_heatmap.png')
#     plt.close()
#     print("Saved: ./feature_detection/harris_heatmap.png")
#
# if __name__ == "__main__":
#     main()

# import os
# import numpy as np
# import scipy.ndimage
# from PIL import Image
#
# # 读取图像函数（灰度）
# def read_img(path):
#     img = Image.open(path).convert('L')  # 转为灰度图
#     return np.array(img).astype(np.float32)
#
# # 保存图像函数（归一化）
# def save_img(img, path):
#     img = img - img.min()
#     img = img / img.max()
#     img = (img * 255).astype(np.uint8)
#     Image.fromarray(img).save(path)
#     print(f"Saved: {path}")
#
# # ✅ 任务一核心函数
# def corner_score(image, u=0, v=0, window_size=(5, 5)):
#     """
#     计算图像中每个像素点的角点响应得分 E(u,v)
#
#     参数:
#     - image: 输入图像（灰度图）
#     - u, v: 偏移量
#     - window_size: 局部求和窗口大小
#
#     返回:
#     - score: 响应图（H x W）
#     """
#
#     # 使用 np.roll 实现图像平移（v 是 y 方向，u 是 x 方向）
#     shifted = np.roll(image, shift=(v, u), axis=(0, 1))
#
#     # 差值的平方项
#     diff_squared = (image - shifted) ** 2
#
#     # 构造卷积窗口（全1的窗口）
#     window = np.ones(window_size)
#
#     # 对每个像素局部区域求和，实现响应图
#     score = scipy.ndimage.convolve(diff_squared, window, mode='constant')
#
#     return score
#
# # ✅ 主程序
# def main():
#     # 读取图像
#     img = read_img('./grace_hopper.png')
#
#     # 创建输出目录
#     if not os.path.exists('./feature_detection'):
#         os.makedirs('./feature_detection')
#
#     # 指定偏移量列表
#     offsets = [(0, 5), (0, -5), (5, 0), (-5, 0)]
#     window_size = (5, 5)
#
#     # 依次计算每个偏移量对应的响应图并保存
#     for u, v in offsets:
#         score = corner_score(img, u=u, v=v, window_size=window_size)
#         filename = f'./feature_detection/corner_score_u{u}_v{v}.png'
#         save_img(score, filename)
#
# if __name__ == "__main__":
#     main()



# import os
#
# import numpy as np
# import scipy.ndimage
# # Use scipy.ndimage.convolve() for convolution.
# # Use zero padding (Set mode = 'constant'). Refer docs for further info.
#
# from PIL import Image
#
# from common import read_img, save_img
#
#
# def corner_score(image, u=5, v=5, window_size=(5, 5)):
#     """
#     Given an input image, x_offset, y_offset, and window_size,
#     return the function E(u,v) for window size W
#     corner detector score for that pixel.
#     Use zero-padding to handle window values outside of the image.
#
#     Input- image: H x W
#            u: a scalar for x offset
#            v: a scalar for y offset
#            window_size: a tuple for window size
#
#     Output- results: a image of size H x W
#     """
#     output = None
#     return output
#
#
# def harris_detector(image, window_size=(5, 5)):
#     """
#     Given an input image, calculate the Harris Detector score for all pixels
#     You can use same-padding for intensity (or 0-padding for derivatives)
#     to handle window values outside of the image.
#
#     Input- image: H x W
#     Output- results: a image of size H x W
#     """
#     # compute the derivatives
#     Ix = None
#     Iy = None
#
#     Ixx = None
#     Iyy = None
#     Ixy = None
#
#     # For each image location, construct the structure tensor and calculate
#     # the Harris response
#     response = None
#
#     return response
#
#
# def main():
#     img = read_img('./grace_hopper.png')
#
#     # Feature Detection
#     if not os.path.exists("./feature_detection"):
#         os.makedirs("./feature_detection")
#
#     # -- TODO Task 1: Corner Score --
#     # (a): Complete corner_score()
#
#     # (b)
#     # Define offsets and window size and calulcate corner score
#     u, v, W = None, None, None
#
#     score = corner_score(img, u, v, W)
#     save_img(score, "./feature_detection/corner_score.png")
#
#     # -- TODO Task 2: Harris Corner Detector --
#     # (a): Complete harris_detector()
#
#     # (b)
#     harris_corners = harris_detector(img)
#     save_img(harris_corners, "./feature_detection/harris_response.png")
#
#
# if __name__ == "__main__":
#     main()