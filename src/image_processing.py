import numpy as np
import scipy.ndimage as ndi
from numba import njit

from constants import FGROUND, BGROUND
import sl


@njit
def label(image):
    """ Connected Component Labeling (4-connected pixels) """

    labels_image = np.zeros_like(image, dtype=np.int64)
    new_label = 1
    label_pixels = [[(0, 0)]]
    imageh, imagew = image.shape
    for y in range(imageh):
        for x in range(imagew):
            if image[y, x] == BGROUND:
                continue

            new_label_assigned = False

            if x == 0:
                labels_image[y, x] = new_label
                label_pixels.append([(y, x)])
                new_label_assigned = True
            else:
                left_neighbor = image[y, x-1]
                if left_neighbor == BGROUND:
                    labels_image[y, x] = new_label
                    label_pixels.append([(y, x)])
                    new_label_assigned = True
                else:
                    left_neighbor_label = labels_image[y, x-1]
                    labels_image[y, x] = left_neighbor_label
                    label_pixels[left_neighbor_label].append((y, x))

            if new_label_assigned:
                new_label += 1

            if y > 0:
                upper_neighbor = image[y-1, x]
                if upper_neighbor == BGROUND:
                    continue

                upper_neighbor_label = labels_image[y-1, x]
                current_pixel_label = labels_image[y, x]
                if upper_neighbor_label == current_pixel_label:
                    continue

                for pixel in label_pixels[current_pixel_label]:
                    labels_image[pixel] = upper_neighbor_label
                label_pixels[upper_neighbor_label].extend(
                    label_pixels[current_pixel_label])

    return labels_image


@njit
def hrlsa(image, threshold):
    """ horizontal Run-length Smearing Algorithm """

    imageh, imagew = image.shape
    for y in range(imageh):
        image_border = True
        adjacent_bground_pixels_counter = 0
        for x in range(imagew):
            if image[y, x] == BGROUND:
                adjacent_bground_pixels_counter += 1
            else:
                if not image_border and adjacent_bground_pixels_counter <= threshold:
                    for i in range(1, adjacent_bground_pixels_counter+1):
                        image[y, x-i] = FGROUND
                image_border = False
                adjacent_bground_pixels_counter = 0
    return image


@njit
def vrlsa(image, threshold):
    """ vertical Run-length Smearing Algorithm """

    return hrlsa(image.T, threshold).T


# def bbox(image):
#     """ minimum bounding box """

#     labels, _ = ndi.label(image)
#     box = None
#     for obj_slice in ndi.find_objects(labels):
#         if box is None:
#             box = obj_slice
#         else:
#             box = sl.union(box, obj_slice)
#     return box, image[box]

def bbox(image):
    """ minimum bounding box """

    ystart, ystop, xstart, xstop = _bbox(image)
    box = slice(ystart, ystop), slice(xstart, xstop)
    return box, image[box]


@njit
def _bbox(image):
    imageh, imagew = image.shape
    box = np.empty(4, dtype=np.int32)
    fground_pixel_found = False
    for y in range(imageh):
        for x in range(imagew):
            if image[y, x] == FGROUND:
                box[0] = y
                fground_pixel_found = True
                break
        if fground_pixel_found:
            break
    fground_pixel_found = False
    for y in range(imageh-1, -1, -1):
        for x in range(imagew):
            if image[y, x] == FGROUND:
                box[1] = y+1
                fground_pixel_found = True
                break
        if fground_pixel_found:
            break
    fground_pixel_found = False
    for x in range(imagew):
        for y in range(imageh):
            if image[y, x] == FGROUND:
                box[2] = x
                fground_pixel_found = True
                break
        if fground_pixel_found:
            break
    fground_pixel_found = False
    for x in range(imagew-1, -1, -1):
        for y in range(imageh):
            if image[y, x] == FGROUND:
                box[3] = x+1
                fground_pixel_found = True
                break
        if fground_pixel_found:
            break
    return box


def project(image, axis, threshold=0):
    """ 
        projection segmentation
        axis = 1 for horizontal, axis = 0 for vertical
    """

    projection = (image == FGROUND).sum(axis=axis)
    fground_column = projection > threshold
    projection_slices = []
    slice_start = 0 if fground_column[0] else None
    for ii in range(len(projection)-1):
        if fground_column[ii] != fground_column[ii+1]:
            if slice_start is None:
                slice_start = ii+1
            else:
                projection_slices.append(slice(slice_start, ii+1))
                slice_start = None
    if slice_start is not None:
        projection_slices.append(slice(slice_start, len(fground_column)))

    axis_slice = slice(0, image.shape[axis])
    segment_locations = []
    for proj_slice in projection_slices:
        if axis == 0:
            segment_loc = axis_slice, proj_slice
        else:
            segment_loc = proj_slice, axis_slice
        segment_locations.append(segment_loc)
    return segment_locations
