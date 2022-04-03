from math import sin, cos, radians

import cv2
import numpy as np


def compute_lines_intersection_take_point(line_1, line_2):
    def get_line_coefficients(line):
        p1, p2 = line
        a = (p1[1] - p2[1])
        b = (p2[0] - p1[0])
        c = (p1[0] * p2[1] - p2[0] * p1[1])
        return a, b, c

    a1, b1, c1 = get_line_coefficients(line_1)
    a2, b2, c2 = get_line_coefficients(line_2)

    c1 *= -1
    c2 *= -1

    d = a1 * b2 - b1 * a2
    dx = c1 * b2 - b1 * c2
    dy = a1 * c2 - c1 * a2
    if abs(d) <= 1e-9:
        return None

    x = dx / d
    y = dy / d
    return [x, y]


def rotate_shape(points, pivot, degrees):
    theta = radians(degrees)
    cosang, sinang = cos(theta), sin(theta)

    cx = pivot.x
    cy = pivot.y

    new_points = []
    for x, y in points:
        tx, ty = x - cx, y - cy
        new_x = (tx * cosang + ty * sinang) + cx
        new_y = (-tx * sinang + ty * cosang) + cy
        new_points.append([new_x, new_y])

    return np.array(new_points)


def distance_line_to_point(line, point):
    return [
        np.linalg.norm([line[0][0] - point[0], line[0][1] - point[1]]),
        np.linalg.norm([line[1][0] - point[0], line[1][1] - point[1]])
    ]


def handle_noise_if_needed(img):
    noisy_pixels_count = np.count_nonzero(np.logical_and(img > 1, img < 255))
    if noisy_pixels_count < 25:
        return img

    kernel = np.ones((3, 3))
    _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    def custom_erosion(img):
        img_res = img.copy()
        w, h = img.shape
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                pixel = img[i, j]
                if pixel != 255:
                    continue

                pixel_area = img[i-1:i+2, j-1:j+2]
                pixel_area[1, 1] = 0
                if not np.count_nonzero(pixel_area):
                    img_res[i, j] = 0
        return img_res

    img = custom_erosion(img)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    return img