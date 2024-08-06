import random
import cv2
import elasticdeform
import numpy as np
from scipy.ndimage import gaussian_filter

def generate_prob_function(mask_shape):
    sigma = np.random.uniform(3, 15)
    a = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1]))
    a_2 = gaussian_filter(a, sigma=sigma)
    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a = scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base
    return a

def get_texture(mask_shape):
    a = generate_prob_function(mask_shape)
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1]))
    b = (a > random_sample).astype(float)
    if np.random.uniform() < 0.7:
        sigma_b = np.random.uniform(3, 5)
    else:
        sigma_b = np.random.uniform(5, 8)
    b2 = gaussian_filter(b, sigma_b)
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b2 > 0.12
    beta = u_0 / (np.sum(b2 * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta * b2, 0, 1)
    return Bj

def get_predefined_texture(mask_shape, sigma_a, sigma_b):
    a = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1]))
    a_2 = gaussian_filter(a, sigma=sigma_a)
    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a = scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1]))
    b = (a > random_sample).astype(float)
    b = gaussian_filter(b, sigma_b)
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta * b, 0, 1)
    return Bj

def random_select(mask_scan):
    if np.sum(mask_scan) == 0:
        return [random.randint(60, 180), random.randint(60, 180)] # 如果mask_scan为空，则随机选择一个点
    y_start, y_end = np.where(np.any(mask_scan, axis=0))[0][[0, -1]]
    y = round(random.uniform(0.3, 0.7) * (y_end - y_start)) + y_start
    # liver_mask = mask_scan[:, y]
    liver_mask = mask_scan
    kernel = np.ones((5, 5), dtype=np.uint8)
    liver_mask = cv2.erode(liver_mask, kernel, iterations=1)
    coordinates = np.argwhere(liver_mask == 1)
    random_index = np.random.randint(0, len(coordinates))
    xy = coordinates[random_index].tolist()
    xy.append(y)
    potential_points = xy
    return potential_points

def get_ellipsoid(x, y):
    """
    根据椭圆的参数生成一个二值图像。
    
    Args:
        x (int): 椭圆的长轴半径。
        y (int): 椭圆的短轴半径。
    
    Returns:
        np.ndarray: 形状为(4*x, 4*y)的二值图像，其中椭圆区域为1，其他区域为0。
    
    """
    sh = (4 * x, 4 * y)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)
    radii = np.array([x, y])
    com = np.array([2 * x, 2 * y])
    bboxl = np.floor(com - radii).clip(0, None).astype(int)
    bboxh = (np.ceil(com + radii) + 1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice, bboxl, bboxh))]
    roiaux = aux[tuple(map(slice, bboxl, bboxh))]
    logrid = map(np.square, np.ogrid[tuple(map(slice, (bboxl - com) / radii, (bboxh - com - 1) / radii, 1j * (bboxh - bboxl)))])
    dst = (1 - sum(logrid)).clip(0, None)
    mask = dst > roiaux
    roi[mask] = 1
    np.copyto(roiaux, dst, where=mask)
    return out

def get_fixed_geo(mask_scan, tumor_type):
    enlarge_x, enlarge_y = 160, 160
    geo_mask = np.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y), dtype=np.int8)
    tiny_radius, small_radius, medium_radius, large_radius = 4, 8, 16, 32

    if tumor_type == 'tiny':
        num_tumor = random.randint(3, 10)
        for _ in range(num_tumor):
            x = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            y = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            sigma = random.uniform(0.5, 1)
            geo = get_ellipsoid(x, y)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            geo_mask[x_low:x_high, y_low:y_high] += geo

    if tumor_type == 'small':
        num_tumor = random.randint(3, 10)
        for _ in range(num_tumor):
            x = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            y = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            sigma = random.randint(1, 2)
            geo = get_ellipsoid(x, y)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            geo_mask[x_low:x_high, y_low:y_high] += geo

    if tumor_type == 'medium':
        num_tumor = random.randint(2, 5)
        for _ in range(num_tumor):
            x = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            y = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            sigma = random.randint(3, 6)
            geo = get_ellipsoid(x, y)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            geo_mask[x_low:x_high, y_low:y_high] += geo

    if tumor_type == 'large':
        num_tumor = random.randint(1, 3)
        for _ in range(num_tumor):
            x = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            y = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            sigma = random.randint(5, 10)
            geo = get_ellipsoid(x, y)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            geo_mask[x_low:x_high, y_low:y_high] += geo

    if tumor_type == "mix":
        num_tumor = random.randint(3, 10)
        for _ in range(num_tumor):
            x = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            y = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            sigma = random.uniform(0.5, 1)
            geo = get_ellipsoid(x, y)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            geo_mask[x_low:x_high, y_low:y_high] += geo

        num_tumor = random.randint(5, 10)
        for _ in range(num_tumor):
            x = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            y = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            sigma = random.randint(1, 2)
            geo = get_ellipsoid(x, y)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            geo_mask[x_low:x_high, y_low:y_high] += geo

        num_tumor = random.randint(2, 5)
        for _ in range(num_tumor):
            x = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            y = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            sigma = random.randint(3, 6)
            geo = get_ellipsoid(x, y)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            geo_mask[x_low:x_high, y_low:y_high] += geo

        num_tumor = random.randint(1, 3)
        for _ in range(num_tumor):
            x = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            y = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            sigma = random.randint(5, 10)
            geo = get_ellipsoid(x, y)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            geo_mask[x_low:x_high, y_low:y_high] += geo

    geo_mask = geo_mask[enlarge_x // 2:-enlarge_x // 2, enlarge_y // 2:-enlarge_y // 2]
    
    # here
    # geo_mask = (geo_mask * mask_scan) >= 1
    geo_mask = geo_mask  >= 1
    return geo_mask

# def get_tumor(volume_scan, mask_scan, tumor_type, texture):
#     geo_mask = get_fixed_geo(mask_scan, tumor_type)
#     sigma = np.random.uniform(1, 2)
#     difference = np.random.uniform(65, 145)
#     geo_blur = gaussian_filter(geo_mask * 1.0, sigma)
#     abnormally = (volume_scan - texture * geo_blur * difference) * mask_scan
#     abnormally_full = volume_scan * (1 - mask_scan) + abnormally
#     abnormally_mask = mask_scan + geo_mask
#     return abnormally_full, abnormally_mask

# def SynthesisTumor(volume_scan, mask_scan, tumor_type, texture):
#     x_start, x_end = np.where(np.any(mask_scan, axis=1))[0][[0, -1]]
#     y_start, y_end = np.where(np.any(mask_scan, axis=0))[0][[0, -1]]
#     x_start, x_end = max(0, x_start + 1), min(mask_scan.shape[0], x_end - 1)
#     y_start, y_end = max(0, y_start + 1), min(mask_scan.shape[1], y_end - 1)
#     liver_volume = volume_scan[:, y_start:y_end]
#     liver_mask = mask_scan[:, y_start:y_end]
#     x_length, y_length = x_end - x_start, y_end - y_start
#     start_x = random.randint(0, texture.shape[0] - x_length - 1)
#     start_y = random.randint(0, texture.shape[1] - y_length - 1)
#     cut_texture = texture[:, start_y:start_y + y_length]
#     liver_volume, liver_mask = get_tumor(liver_volume, liver_mask, tumor_type, cut_texture)
#     volume_scan[x_start:x_end, y_start:y_end] = liver_volume
#     mask_scan[x_start:x_end, y_start:y_end] = liver_mask
#     return volume_scan, mask_scan

def get_tumor(volume_scan, mask_scan, tumor_type, texture):
    geo_mask = get_fixed_geo(mask_scan, tumor_type)
    
    sigma = np.random.uniform(1, 2)
    # difference = np.random.uniform(65, 145)
    # geo_blur = gaussian_filter(geo_mask * 1.0, sigma)
    # abnormally = (volume_scan - texture * geo_blur * difference) * mask_scan
    # abnormally_full = volume_scan * (1 - mask_scan) + abnormally
    # abnormally_mask = mask_scan + geo_mask
    difference = np.random.uniform(65/255, 145/255)
    # difference = np.random.uniform(1/255, 255/255)
    geo_blur = gaussian_filter(geo_mask * 1.0, sigma)
    if np.sum(geo_blur) == 0:
        return volume_scan, np.zeros_like(mask_scan)
    # abnormally = (volume_scan - texture * geo_blur * difference) * geo_mask
    abnormally = (  texture * geo_blur * difference) * geo_mask
    abnormally_full = volume_scan * (1 - geo_mask)  + abnormally
    abnormally_mask = geo_mask
    return abnormally_full, abnormally_mask

from matplotlib import pyplot as plt

def SynthesisTumor(volume_scan, mask_scan, tumor_type, texture):
    x_start, x_end = np.where(np.any(mask_scan, axis=1))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask_scan, axis=0))[0][[0, -1]]
    x_start, x_end = max(0, x_start + 1), min(mask_scan.shape[0], x_end - 1)
    y_start, y_end = max(0, y_start + 1), min(mask_scan.shape[1], y_end - 1)
    liver_volume = volume_scan[x_start:x_end, y_start:y_end]
    liver_mask = mask_scan[x_start:x_end, y_start:y_end]
    x_length, y_length = x_end - x_start, y_end - y_start

    # Ensure that cut_texture has the same dimensions as liver_volume
    cut_texture = texture[:x_length, :y_length]

    liver_volume, liver_mask = get_tumor(liver_volume, liver_mask, tumor_type, cut_texture)
    volume_scan[x_start:x_end, y_start:y_end] = liver_volume
    mask_scan = np.zeros_like(mask_scan)
    mask_scan[x_start:x_end, y_start:y_end] = liver_mask
    volume_scan[volume_scan <0] = 0
    # plt.imshow(volume_scan)
    # plt.show()
    # plt.imshow(mask_scan)
    # plt.show()
    return volume_scan, mask_scan