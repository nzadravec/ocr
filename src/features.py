import numpy as np
import scipy.ndimage as ndi
from numba import njit

from constants import FGROUND, BGROUND


def gsc_features(char_image, patches_per_dim_number=4, orientations_number=12):
    _FGROUND = 1
    _BGROUND = 0
    im = np.where(char_image == BGROUND, _BGROUND, _FGROUND).astype(np.int32)

    Gx = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=np.int32
    )
    Gy = np.array(
        [[-1, -2, -1],
         [0,  0,  0],
         [1,  2,  1]],
        dtype=np.int32
    )

    gx_image = ndi.convolve(im, Gx, mode='constant', cval=_BGROUND)
    gy_image = ndi.convolve(im, Gy, mode='constant', cval=_BGROUND)

    return gsc_features_(im, _FGROUND, _BGROUND, gx_image, gy_image, patches_per_dim_number=patches_per_dim_number, orientations_number=orientations_number)


@njit
def gsc_features_(char_image, _FGROUND, _BGROUND, gx_image, gy_image, patches_per_dim_number=4, orientations_number=12):
    nearest_neighbors_shift = [
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1)
    ]

    structural_rules = [
        ((0, [2, 3, 4]), (4, [2, 3, 4])),
        ((0, [8, 9, 10]), (4, [8, 9, 10])),
        ((2, [5, 6, 7]), (6, [5, 6, 7])),
        ((2, [1, 0, 11]), (6, [1, 0, 11])),
        ((5, [4, 5, 6]), (1, [4, 5, 6])),
        ((5, [10, 11, 0]), (1, [10, 11, 0])),
        ((3, [1, 2, 3]), (7, [1, 2, 3])),
        ((3, [7, 8, 9]), (7, [7, 8, 9])),
        ((2, [5, 6, 7]), (0, [8, 9, 10])),
        ((6, [5, 6, 7]), (0, [2, 3, 4])),
        ((4, [8, 9, 10]), (2, [1, 0, 11])),
        ((6, [1, 0, 11]), (4, [2, 3, 4])),
    ]

    orientations_mapper = [
        0,
        11,
        10,
        9,
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
    ]

    for rule_idx, (neighbor1_rule, neighbor2_rule) in enumerate(structural_rules):
        _, n1_orientation_range = neighbor1_rule
        for i in range(len(n1_orientation_range)):
            n1_orientation_range[i] = orientations_mapper[n1_orientation_range[i]]
        _, n2_orientation_range = neighbor2_rule
        for i in range(len(n2_orientation_range)):
            n2_orientation_range[i] = orientations_mapper[n2_orientation_range[i]]

    image_shape = char_image.shape
    patch_shape = image_shape[0]//patches_per_dim_number, image_shape[1]//patches_per_dim_number

    grad_direction_image = np.arctan2(gy_image, gx_image)
    for yy in range(image_shape[0]):
        for xx in range(image_shape[1]):
            if grad_direction_image[yy, xx] < 0:
                grad_direction_image[yy, xx] += 2*np.pi

    grad_orientation_image = (grad_direction_image//(
        2*np.pi/orientations_number)).astype(np.int32)
    for yy in range(image_shape[0]):
        for xx in range(image_shape[1]):
            if gx_image[yy, xx] == 0 and gy_image[yy, xx] == 0:
                grad_orientation_image[yy, xx] = -1

    structural_rule_images = np.full(
        (len(structural_rules), image_shape[0], image_shape[1]), 0, dtype=np.int32)
    for rule_idx, (neighbor1_rule, neighbor2_rule) in enumerate(structural_rules):
        structural_rule_image = structural_rule_images[rule_idx]
        n1_nn_shift_idx, n1_orientation_range = neighbor1_rule
        n2_nn_shift_idx, n2_orientation_range = neighbor2_rule
        n1_dy, n1_dx = nearest_neighbors_shift[n1_nn_shift_idx]
        n2_dy, n2_dx = nearest_neighbors_shift[n2_nn_shift_idx]
        for yy in range(image_shape[0]):
            for xx in range(image_shape[1]):
                n1_yy = yy + n1_dy
                n1_xx = xx + n1_dx
                n2_yy = yy + n2_dy
                n2_xx = xx + n2_dx
                if not (0 <= n1_yy < image_shape[0] and 0 <= n1_xx < image_shape[1] and 0 <= n2_yy < image_shape[0] and 0 <= n2_xx < image_shape[1]):
                    continue
                if grad_orientation_image[n1_yy, n1_xx] in n1_orientation_range and grad_orientation_image[n2_yy, n2_xx] in n2_orientation_range:
                    structural_rule_image[yy, xx] = 1

    concavity_features_patch_number = 8
    concavity_images = np.full(
        (5, image_shape[0], image_shape[1]), 0, dtype=np.int32)
    bground_pixels = set()  # reached from any orientation
    for yy in range(image_shape[0]):
        for xx in range(image_shape[1]):
            if char_image[yy, xx] == _FGROUND:
                break
            concavity_images[0, yy, xx] = 1
            bground_pixels.add((yy, xx))
    for xx in range(image_shape[1]):
        for yy in range(image_shape[0]):
            if char_image[yy, xx] == _FGROUND:
                break
            concavity_images[1, yy, xx] = 1
            bground_pixels.add((yy, xx))
    for yy in range(image_shape[0]):
        for xx in range(image_shape[1]-1, -1, -1):
            if char_image[yy, xx] == _FGROUND:
                break
            concavity_images[2, yy, xx] = 1
            bground_pixels.add((yy, xx))
    for xx in range(image_shape[1]):
        for yy in range(image_shape[0]-1, -1, -1):
            if char_image[yy, xx] == _FGROUND:
                break
            concavity_images[3, yy, xx] = 1
            bground_pixels.add((yy, xx))
    for yy in range(image_shape[0]):
        for xx in range(image_shape[1]):
            if char_image[yy, xx] == _BGROUND and (yy, xx) not in bground_pixels:
                concavity_images[4, yy, xx] = 1

    pixels_in_patch_number = patch_shape[0] * patch_shape[1]
    gradient_features = np.zeros(
        patches_per_dim_number*patches_per_dim_number*orientations_number, dtype=np.float32)
    structural_features = np.zeros(
        patches_per_dim_number*patches_per_dim_number*len(structural_rules), dtype=np.float32)
    concavity_features = np.zeros(
        patches_per_dim_number*patches_per_dim_number*concavity_features_patch_number, dtype=np.float32)
    patch_index = 0
    patch_image = np.empty((patches_per_dim_number, patches_per_dim_number,
                            patch_shape[0], patch_shape[1]), dtype=np.int32)
    for yy, ystart in enumerate(range(0, image_shape[0], patch_shape[0])):
        for xx, xstart in enumerate(range(0, image_shape[1], patch_shape[1])):
            patch = char_image[ystart:ystart +
                               patch_shape[0], xstart:xstart+patch_shape[1]]
            patch_image[yy, xx] = patch

            # gradient features
            gradient_features_patch = gradient_features[orientations_number *
                                                        patch_index:orientations_number * (patch_index+1)]
            grad_orientation_image_patch = grad_orientation_image[ystart:ystart +
                                                                  patch_shape[0], xstart:xstart+patch_shape[1]]
            for jj in range(patch_shape[0]):
                for ii in range(patch_shape[1]):
                    grad_orientation = grad_orientation_image_patch[jj, ii]
                    if patch[jj, ii] == _FGROUND and grad_orientation != -1:
                        gradient_features_patch[grad_orientation] += 1
            gradient_features_patch /= pixels_in_patch_number

            # structural features
            structural_features_patch = structural_features[len(structural_rules) *
                                                            patch_index:len(structural_rules) * (patch_index+1)]
            structural_rule_images_patch = structural_rule_images[:, ystart:ystart +
                                                                  patch_shape[0], xstart:xstart+patch_shape[1]]
            for rule_idx in range(len(structural_rules)):
                structural_rule_image_patch = structural_rule_images_patch[rule_idx]
                count = 0
                for jj in range(patch_shape[0]):
                    for ii in range(patch_shape[1]):
                        if patch[jj, ii] == _FGROUND:
                            count += structural_rule_image_patch[jj, ii]
                structural_features_patch[rule_idx] = count / \
                    pixels_in_patch_number

            # concavity features
            concavity_features_patch = concavity_features[concavity_features_patch_number *
                                                          patch_index:concavity_features_patch_number * (patch_index+1)]
            concavity_features_patch[0] = (
                patch == _FGROUND).sum() / pixels_in_patch_number
            max_horizontal_stroke_length = 0
            for jj in range(patch_shape[0]):
                horizontal_stroke_length = 0
                for ii in range(patch_shape[1]):
                    if patch[jj, ii] == _FGROUND:
                        horizontal_stroke_length += 1
                    else:
                        max_horizontal_stroke_length = max(
                            max_horizontal_stroke_length, horizontal_stroke_length)
                        horizontal_stroke_length = 0
                max_horizontal_stroke_length = max(
                    max_horizontal_stroke_length, horizontal_stroke_length)
            concavity_features_patch[1] = max_horizontal_stroke_length / \
                patch_shape[1]
            max_vertical_stroke_length = 0
            for ii in range(patch_shape[1]):
                vertical_stroke_length = 0
                for jj in range(patch_shape[0]):
                    if patch[jj, ii] == _FGROUND:
                        vertical_stroke_length += 1
                    else:
                        max_vertical_stroke_length = max(
                            max_vertical_stroke_length, vertical_stroke_length)
                        vertical_stroke_length = 0
                max_vertical_stroke_length = max(
                    max_vertical_stroke_length, vertical_stroke_length)
            concavity_features_patch[2] = max_vertical_stroke_length / \
                patch_shape[0]
            concavity_images_patch = concavity_images[:, ystart:ystart +
                                                      patch_shape[0], xstart:xstart+patch_shape[1]]
            for i in range(5):
                concavity_features_patch[i+3] = concavity_images_patch[i].sum(
                ) / pixels_in_patch_number

            patch_index += 1

    return np.concatenate((gradient_features, structural_features, concavity_features))
