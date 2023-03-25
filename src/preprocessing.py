import numpy as np
import scipy.ndimage as ndi
import scipy.stats as stats
import skimage.filters as filters
from numba import njit
from numba.typed import List

from constants import FGROUND, BGROUND
from image_processing import hrlsa


def rgb2gray(rgb):
    # L = R * 299/1000 + G * 587/1000 + B * 114/1000
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def threshold_otsu(image, bins_number=256):
    """ Otsu's thresholding method (https://en.wikipedia.org/wiki/Otsu%27s_method) """

    xs = np.arange(bins_number)
    hs, _ = np.histogram(image, bins=bins_number, density=True)
    varbs = np.zeros(bins_number)
    for ii in range(1, bins_number):
        p0 = np.sum(hs[:ii])
        p1 = 1 - p0

        mu0 = np.sum(xs[:ii]*hs[:ii]) / p0
        mu1 = np.sum(xs[ii:]*hs[ii:]) / p1
        varb = p0 * p1 * (mu0 - mu1) ** 2
        varbs[ii] = varb

    maxvarb_indices = np.argwhere(varbs == np.max(varbs)).flatten()
    median_idx = maxvarb_indices[len(maxvarb_indices)//2]

    return xs[median_idx]


def binarize(image, threshold=filters.threshold_otsu):
    if image.dtype == np.float32:
        image *= 255
        image = image.astype(np.uint8)
    if image.ndim >= 3:
        image = rgb2gray(image)
    if image.dtype == np.uint8:
        gray_levels = np.unique(image)
        if len(gray_levels) > 2:
            t = threshold(image)
        else:
            t = max(gray_levels)
        image = np.where(image < t, FGROUND, BGROUND).astype(bool)
    return image


# Implementation of 'Border noise removal' from
# A simple and effective approach for border noise removal from document images (https://ieeexplore.ieee.org/document/5383115)


def remove_border_noise(
        image, window_shape=(5, 5), step=(5, 5), fground_margin_ratio=1/6, fground_threshold=0.7, concomp_margin=25, bground_threshold=0.995):

    fground_margin_ratio = [fground_margin_ratio]*4

    concomp_margin = [concomp_margin]*4

    bground_margin_ratio = [1/10, 1/10, 1/25, 1/50]

    border_pixel_filter(
        image, FGROUND, fground_margin_ratio, fground_threshold, window_shape, step)
    border_concomp_removal(image, concomp_margin)
    border_pixel_filter(
        image, BGROUND, bground_margin_ratio, bground_threshold, window_shape, step)


def border_concomp_removal(image, margin):
    imageh, imagew = image.shape

    labels, _ = ndi.label(image)
    obj_label = 0
    for obj_location in ndi.find_objects(labels):
        obj_label += 1

        if obj_location == None:
            continue

        yslice, xslice = obj_location
        if (yslice.start < margin[2] or
            yslice.stop > imageh-margin[3] or
            xslice.start < margin[0] or
                xslice.stop > imagew-margin[1]):
            image[labels == obj_label] = BGROUND


def border_pixel_filter(image, pixel_type, margin_ratio, threshold, window_shape=(5, 5), step=(5, 5)):
    imageh, imagew = image.shape

    window_width = window_shape[0]
    xstep = step[0]
    window_height = window_shape[1]
    ystep = step[1]
    top_bottom_xstart = 0
    top_bottom_xstop = imagew

    # left & right margin

    window_area = imageh * window_width
    for x in range(round(margin_ratio[0]*imagew), 0, -xstep):
        image_window_region = image[:, max(0, x-window_width):x]
        pixel_type_num = (image_window_region == pixel_type).sum()
        ratio = pixel_type_num / window_area
        if ratio > threshold:
            image[:, :x] = BGROUND
            top_bottom_xstart = x
            break

    for x in range(round(imagew-margin_ratio[1]*imagew), imagew, xstep):
        image_window_region = image[:, x:min(imagew, x+window_width)]
        pixel_type_num = (image_window_region == pixel_type).sum()
        ratio = pixel_type_num / window_area
        if ratio > threshold:
            image[:, x:] = BGROUND
            top_bottom_xstop = x
            break

    # top & bottom margin

    top_bottom_xslice = slice(top_bottom_xstart, top_bottom_xstop)

    window_area = window_height * (top_bottom_xstop - top_bottom_xstart)
    for y in range(round(margin_ratio[2]*imageh), 0, -ystep):
        window_yslice = slice(max(0, y-window_height), y)
        image_window_region = image[window_yslice, top_bottom_xslice]
        pixel_type_num = (image_window_region == pixel_type).sum()
        ratio = pixel_type_num / window_area
        if ratio > threshold:
            image[:y, :] = BGROUND
            break

    for y in range(round(imageh-margin_ratio[3]*imageh), imageh, ystep):
        window_yslice = slice(y, min(imageh, y+window_height))
        image_window_region = image[window_yslice, top_bottom_xslice]
        pixel_type_num = (image_window_region == pixel_type).sum()
        ratio = pixel_type_num / window_area
        if ratio > threshold:
            image[y:, :] = BGROUND
            break


# Implementation of 'Skew detection' from
# Skew detection and block classification of printed documents (https://www.sciencedirect.com/science/article/abs/pii/S0262885600000986)


def correct_skew(image):
    skew_angle = detect_skew(image)
    return ndi.rotate(image, -skew_angle, cval=BGROUND, order=0, prefilter=False)


def detect_skew(image):
    T = 20
    smoothed_image = hrlsa(image.copy(), T)

    def fground_bground_transitions(iline, oline):
        np.logical_and(iline[:-1] == FGROUND, iline[1:] == BGROUND, out=oline)
    burst_image = ndi.generic_filter1d(
        smoothed_image.T, fground_bground_transitions, 2, mode='constant', cval=0, origin=-1).T

    fground_pixels_on_vlines = List()
    for vline in _generate_vertical_lines(burst_image.shape[1]):
        fground_pixels_on_vline = np.array(
            [(y, vline) for y in np.where(burst_image[:, vline] == FGROUND)[0]])
        if len(fground_pixels_on_vline) == 0:
            continue
        fground_pixels_on_vlines.append(fground_pixels_on_vline)

    return _oned_hough_transform(fground_pixels_on_vlines)


def _generate_vertical_lines(width, line_num=30):
    xs, step = np.linspace(0, width, num=line_num+2, retstep=True)
    xs = xs[1:-1]
    dxs = stats.truncnorm(-step/2, step/2, loc=0, scale=step/4).rvs(len(xs))
    return np.rint(xs + dxs).astype(np.int32)


@njit
def _oned_hough_transform(fground_pixels_on_vlines):
    angle_counter = np.zeros(1800)
    for ii in range(len(fground_pixels_on_vlines)-1):
        ps_i = fground_pixels_on_vlines[ii]
        for jj in range(ii+1, len(fground_pixels_on_vlines)):
            ps_j = fground_pixels_on_vlines[jj]
            for (y1, x1) in ps_i:
                for (y2, x2) in ps_j:
                    dy = y2 - y1
                    dx = x2 - x1
                    angle = round(np.degrees(np.arctan2(dy, dx)), 1)
                    angle = min(max(round((angle + 90) * 10), 0), 1800)
                    angle_counter[angle] += 1
    angle = angle_counter.argmax()
    return (-(angle - 900) / 10)


@njit
def remove_salt_pepper(image):
    h, w = image.shape

    for y in range(h):
        for x in range(w):
            if image[y, x] == FGROUND:
                if (image[max(y-1, 0):y+2, max(x-1, 0):x+2] == FGROUND).sum() == 1:
                    image[y, x] = BGROUND

            if image[y, x] == BGROUND:
                if (image[max(y-1, 0):y+2, max(x-1, 0):x+2] == BGROUND).sum() == 1:
                    image[y, x] = FGROUND
