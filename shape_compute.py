import cv2
import numpy as np

from intersection_model import IntersectionModel

# class HoughSettings:
#     def __init__(self):
#         self.rho = 0.5
#         self.theta = np.pi / 360
#         self.threshold = 20
#         self.min_line_length = 9
#         self.max_line_gap = 3


class HoughSettings:
    def __init__(self):
        self.rho = 0.5
        self.theta = np.pi / 360
        self.threshold = 17
        self.min_line_length = 9
        self.max_line_gap = 3


HOUGH_SETTINGS = HoughSettings()


def compute_candidate_contours(img):
    contours_with_children, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_with_children = np.array(contours_with_children, dtype=object)[hierarchy[0][:, 2] != -1]
    return contours_with_children


def compute_shape_contour(img_shape, contour):
    res = np.zeros(img_shape, np.uint8)
    cv2.drawContours(res, [contour], 0, 255, 1)
    lines = np.asarray(
        cv2.HoughLinesP(
            res,
            HOUGH_SETTINGS.rho,
            HOUGH_SETTINGS.theta,
            HOUGH_SETTINGS.threshold,
            minLineLength=HOUGH_SETTINGS.min_line_length,
            maxLineGap=HOUGH_SETTINGS.max_line_gap
        )
    )[:, 0].reshape(-1, 2, 2)

    if len(lines) < 3:
        return []

    lines_lengths = [np.linalg.norm(p1 - p2) for p1, p2 in lines]

    intersection_model = IntersectionModel(lines)\
        .compute_all_intersections()\
        .clear_invalid_intersections()

    shape_points = recursive_contour_search([], intersection_model, lines_lengths, level=0)
    if len(shape_points) < 3:
        return []

    return shape_points


def recursive_contour_search(res, intersection_model, lines_lengths, level):
    model = intersection_model.model
    max_dist = 0
    idx = -1
    for i, intersection_element in enumerate(model[level]):
        next_point = intersection_element[1]
        if next_point in res:
            continue

        if len(res) == 0:
            res.append(intersection_element[1])
            recursive_contour_search(res, intersection_model, lines_lengths, intersection_element[0])
            return res

        dist = np.linalg.norm([next_point[0] - res[-1][0], next_point[1] - res[-1][1]])
        if dist > max_dist:
            max_dist = dist
            idx = i

    if idx == -1 or max_dist < (lines_lengths[level] / 2):
        return res

    intersection_element = model[level][idx]
    res.append(intersection_element[1])
    recursive_contour_search(res, intersection_model, lines_lengths, intersection_element[0])
    return res
