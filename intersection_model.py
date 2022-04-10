import numpy as np

from util import compute_lines_intersection_take_point, distance_line_to_point


class IntersectionModel:
    def __init__(self, lines):
        self.lines = lines
        self.model = [[] for _ in range(len(lines))]

    def _try_add_intersection_point_to_model(self, line_1, idx_1, line_2, idx_2):
        intersection_point = compute_lines_intersection_take_point(line_1, line_2)
        if intersection_point is None:
            return

        d_1 = distance_line_to_point(line_1, intersection_point)
        d_2 = distance_line_to_point(line_2, intersection_point)
        self.model[idx_1].append([idx_2, intersection_point, d_1])
        self.model[idx_2].append([idx_1, intersection_point, d_2])

    def compute_all_intersections(self):
        for i, line_1 in enumerate(self.lines):
            for j, line_2 in enumerate(self.lines[i + 1:]):
                self._try_add_intersection_point_to_model(
                    line_1, i,
                    line_2, i + j + 1
                )

        return self

    def clear_invalid_intersections(self):
        for i, line_intersections in enumerate(self.model):
            if len(line_intersections) <= 2:
                continue

            ordered_p1 = list(np.array(sorted(line_intersections, key=lambda x: x[2][0]), dtype=object))
            ordered_p2 = list(np.array(sorted(line_intersections, key=lambda x: x[2][1]), dtype=object))
            self.model[i] = [ordered_p1[0], ordered_p2[0]]

        return self
