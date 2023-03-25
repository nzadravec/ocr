import numpy as np
import scipy.ndimage as ndi
import sl

from constants import FGROUND, BGROUND
import image_processing as ip


class TextLineSegmenter:
    """ Segments text lines from page image """

    def locate_text_lines(self, page):
        """ page → text line locations 

        Arguments
            page: 2D np.array(bool)

        Return values
            text line locations: list of slices
        """
        raise NotImplementedError

    def segment_text_lines(self, page):
        """ page → text line images 

        Arguments
            page: 2D np.array(bool)

        Return values
            text line images: list of 2D np.array(bool)
        """
        line_locations = self.locate_text_lines(page)
        return [page[loc] for loc in line_locations]


class HorizontalProjection(TextLineSegmenter):

    def __init__(self, threshold=0):
        self.threshold = threshold

    def locate_text_lines(self, page):
        return ip.project(page, axis=1, threshold=self.threshold)


class RunLengthSmearing(TextLineSegmenter):
    """
        Implementation of 'Block segmentation and text discrimination' from
        Document Analysis System (https://ieeexplore.ieee.org/document/5390486)
    """

    def __init__(self, hthreshold=200, vthreshold=200, additional_hthreshold=15):
        """ default values are tested on image resolution of 150dpi """
        self.hthreshold = hthreshold
        self.vthreshold = vthreshold
        self.additional_hthreshold = additional_hthreshold

    def locate_text_lines(self, page):
        zone_locations = self.locate_zones(page)
        return self.filter_text_zones(page, zone_locations)

    def locate_zones(self, page):
        hsmoothed_page = ip.hrlsa(page.copy(), self.hthreshold)
        vsmoothed_page = ip.vrlsa(page.copy(), self.vthreshold)
        hvsmoothed_page = np.where(np.logical_and(
            hsmoothed_page == FGROUND, vsmoothed_page == FGROUND), FGROUND, BGROUND)
        final_page = ip.hrlsa(
            hvsmoothed_page, self.additional_hthreshold)

        labels = ip.label(final_page)
        return [loc for loc in ndi.find_objects(labels) if loc is not None]

    def filter_text_zones(self, page, zone_locations):
        _zone_locations = zone_locations
        zone_locations = []
        features = []
        for zone_loc in _zone_locations:
            zone = page[zone_loc]

            H = zone.shape[0]  # The height of zone

            # Total number of black pixels
            DC = (zone == FGROUND).sum()

            # Horizontal white-black transitions
            def horizontal_bground_fground_transitions(iline, oline):
                np.logical_and(iline[:-1] == BGROUND,
                               iline[1:] == FGROUND, out=oline)
            bground_fground_transitions = ndi.generic_filter1d(
                zone, horizontal_bground_fground_transitions, 2, mode='constant', cval=0)
            TC = bground_fground_transitions.sum()

            if TC == 0:
                continue

            R = DC/TC  # The mean horizontal length of the black runs

            zone_locations.append(zone_loc)
            features.append((H, R))

        medianH = np.median([H for H, _ in features])
        medianR = np.median([R for _, R in features])

        text_zone_locations = []
        for zone_loc, features in zip(zone_locations, features):
            H, R = features
            if not (R < 3 * medianR and medianH//3 < H < 3 * medianH):
                continue

            yslice, xslice = zone_loc
            zoneh = yslice.stop - yslice.start
            zonew = xslice.stop - xslice.start
            if zonew/zoneh < 2.5:
                continue

            text_zone_locations.append(zone_loc)

        return text_zone_locations


class RecursiveXYCutWithHProjection(TextLineSegmenter):

    def __init__(self, threshold=20, maxdepth=None, hprojection_threshold=0):
        self.threshold = threshold
        self.maxdepth = maxdepth
        self.locate_text_lines = HorizontalProjection(
            threshold=hprojection_threshold).locate_text_lines

    def locate_text_lines(self, page):
        zone_locations = self.recursive_xy_cut(page)
        line_locations = []
        for zone_loc in zone_locations:
            line_locations.extend([sl.shift(line_loc, zone_loc)
                                   for line_loc in self.locate_text_lines(page[zone_loc])])
        return line_locations

    def recursive_xy_cut(self, image, _slice=None, depth=0):
        """
            Implementation of 'Recursive X-Y Cut Algorithm' from
            Document and Content Analysis course (https://sites.google.com/a/iupr.com/dia-course/lectures/lecture-document-image-analysis)
        """
        if _slice is None:
            image_slice = image
        else:
            image_slice = image[_slice]

        if self.maxdepth is not None and depth > self.maxdepth:
            return []

        h, w = image_slice.shape
        if h == 0 or w == 0:
            return []

        maxvy, maxvystart, maxvystop = self._find_largest_zero_valley(
            image_slice, axis=1)

        maxvx, maxvxstart, maxvxstop = self._find_largest_zero_valley(
            image_slice, axis=0)

        if max(maxvy, maxvx) < self.threshold:
            return [_slice]

        h, w = image_slice.shape
        if _slice is None:
            ystart = 0
            xstart = 0
        else:
            ystart = _slice[0].start
            xstart = _slice[1].start
        if maxvy > maxvx:
            slice1 = slice(ystart+0, ystart +
                           maxvystart), slice(xstart+0, xstart+w)
            slice2 = slice(ystart+maxvystop, ystart +
                           h), slice(xstart+0, xstart+w)
        else:
            slice1 = slice(ystart+0, ystart+h), slice(xstart +
                                                      0, xstart+maxvxstart)
            slice2 = slice(ystart+0, ystart+h), slice(xstart +
                                                      maxvxstop, xstart+w)
        return self.recursive_xy_cut(image, _slice=slice1, depth=depth+1) + self.recursive_xy_cut(image, _slice=slice2, depth=depth+1)

    def _find_largest_zero_valley(self, image, axis):
        proj = np.sum(image == FGROUND, axis=axis)
        maxvalley = 0
        zero_counter = 0
        vstart = None
        for ii in range(len(proj)):
            if proj[ii] == 0:
                zero_counter += 1
                if vstart is None:
                    vstart = ii
            else:
                if zero_counter > maxvalley:
                    maxvalley = zero_counter
                    maxvstart = vstart
                    maxvstop = ii
                zero_counter = 0
                vstart = None
        if zero_counter > maxvalley:
            maxvalley = zero_counter
            maxvstart = vstart
            maxvstop = ii

        if maxvalley == 0:
            return maxvalley, None, None

        return maxvalley, maxvstart, maxvstop
