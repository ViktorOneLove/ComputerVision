from math import radians

import cv2
import numpy as np
from shapely.geometry import Polygon

from util import rotate_shape


class TransformSearcher:
    def __init__(self, base_shapes, img_contour, centroid, area):
        self.base_shapes = base_shapes
        self.img_contour = img_contour
        self.centroid = centroid
        self.area = area

    def _search_angle(self, transformed_shape):
        best_fitting_angle = None
        max_iou = 0
        transformed_polygon = Polygon(transformed_shape)

        for angle in range(0, 360):
            rotated_shape = rotate_shape(transformed_shape, transformed_polygon.centroid, angle)
            img_with_rotated_shape = np.zeros_like(self.img_contour)
            cv2.drawContours(img_with_rotated_shape, [np.array(rotated_shape).astype(int)], 0, 255, -1)

            iou = np.count_nonzero(np.logical_and(img_with_rotated_shape, self.img_contour)) / np.count_nonzero(
                np.logical_or(img_with_rotated_shape, self.img_contour))

            if iou > max_iou:
                max_iou = iou
                best_fitting_angle = angle

        if max_iou < 0.9:
            return None

        return best_fitting_angle

    def _compute_shifts_w_rotation(self, base_shape, scale, angle):
        res_shape = base_shape * scale
        tmp = res_shape.copy()
        for i in [0, 1]:
            res_shape[:, i] = np.cos(radians(angle)) * tmp[:, i] \
                              - ((-1) ** i) * np.sin(radians(angle)) * tmp[:, 1 - i]
        res_polygon = Polygon(res_shape)
        dx = self.centroid[0] - res_polygon.centroid.x
        dy = self.centroid[1] - res_polygon.centroid.y
        return dx, dy

    def search_transformed_shapes(self):
        for count, base_shape in enumerate(self.base_shapes):
            scale = np.sqrt(self.area / Polygon(base_shape).area)
            transformed_shape = base_shape * scale

            transformed_polygon = Polygon(transformed_shape)
            dx = self.centroid[0] - transformed_polygon.centroid.x
            dy = self.centroid[1] - transformed_polygon.centroid.y
            transformed_shape[:, 0] += dx
            transformed_shape[:, 1] += dy

            angle = self._search_angle(transformed_shape)
            if angle is None:
                continue

            angle = 360 - angle
            dx, dy = self._compute_shifts_w_rotation(base_shape, scale, angle)
            return np.array([count, dx, dy, scale, angle]).astype(int)

        return None
