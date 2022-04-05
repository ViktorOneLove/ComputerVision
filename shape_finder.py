import argparse

import cv2
import numpy as np

from shape_compute import compute_candidate_contours, compute_shape_contour
from transform_searcher import TransformSearcher
from util import handle_noise_if_needed


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help='source image')
    parser.add_argument('-s', '--shape', help='input .txt file with shapes')
    return parser.parse_args()


def parse_input_txt(path):
    with open(path, "r") as f:
        content = f.read().splitlines()

    base_shapes = []
    num_base_shapes = int(content[0])
    for i in range(num_base_shapes):
        shape = []
        coordinates = content[i + 1].split(",")
        for j in range(0, len(coordinates), 2):
            shape.append([int(it) for it in coordinates[j:j+2]])

        base_shapes.append(np.array(shape))

    return base_shapes


def print_results_params_to_cli(result_params_arr):
    print(len(result_params_arr))
    for res_params in result_params_arr:
        print(', '.join(str(it) for it in res_params))


def main():
    args = parse_arguments()

    base_shapes = parse_input_txt(args.shape)
    img_src = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    img_src = handle_noise_if_needed(img_src)

    result_params = []

    candidate_contours = compute_candidate_contours(img_src)

    for contour in candidate_contours:
        shape_contour = compute_shape_contour(img_src.shape, contour)
        if not shape_contour:
            continue

        img_contour = np.zeros_like(img_src)
        cv2.drawContours(img_contour, [np.array(shape_contour).astype(int)], 0, 255, -1)

        _, _, stats, centroids = cv2.connectedComponentsWithStats(img_contour, 8, cv2.CV_32S)
        area = stats[1][cv2.CC_STAT_AREA]
        centroid = centroids[1]

        transformed_shapes = TransformSearcher(base_shapes, img_contour, centroid, area).search_transformed_shapes()
        if transformed_shapes is not None:
            result_params.append(transformed_shapes)

    print_results_params_to_cli(result_params)


if __name__ == '__main__':
    main()
