import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numba import njit

from constants import FGROUND, BGROUND
import image_processing as ip
import sl


class TextLineRecognizer:
    """ Recognizes text from text line image """

    def recognize_text_line(self, text_line):
        """ text_line → str

        Arguments
            text_line: 2D np.array(bool)

        Return values
            text: str
        """
        raise NotImplementedError


class CharacterClassifier:
    """ features (1d array) → class[, probability] """

    def classify(self, x, return_prob=False):
        raise NotImplementedError


class TextLineCharFeatureExtractor:
    """ Extracts text line character features

    Method set_text_line needs to called with text line image whose characters we want to extract features from.
    Character features are obtained by calling extract_features with character image and character location.
    """

    def set_text_line(self, text_line):
        raise NotImplementedError

    def extract_features(self, char, char_location):
        raise NotImplementedError


class CharFeatureExtractor(TextLineCharFeatureExtractor):

    def __init__(self, extract_char_features):
        self.extract_char_features = extract_char_features

    def set_text_line(self, text_line):
        pass

    def extract_features(self, char, char_location):
        return self.extract_char_features(ip.bbox(char)[1])


class ExtractEncodeResize(TextLineCharFeatureExtractor):

    def __init__(self, extract_char_features):
        self.extract_char_features = extract_char_features
        self.debug = False

    def set_text_line(self, text_line):
        self.text_line = text_line
        baseline_ypos = estimate_baseline_yposition(text_line)
        meanline_ypos = estimate_meanline_yposition(text_line)
        self.x_height = baseline_ypos-meanline_ypos

    def extract_features(self, char, char_location):
        if self.debug:
            _, ax = plt.subplots()
            plt.imshow(self.text_line)
            (y, x), w, h = sl.start(char_location), sl.width(
                char_location), sl.height(char_location)
            rect = mpatches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.show()
        box, char = ip.bbox(char)
        char_location = sl.shift(box, sl.start(char_location))
        yslice, _ = char_location
        charh, charw = char.shape
        y_position = yslice.stop/self.x_height
        height = charh/self.x_height
        char_features = self.extract_char_features(char)
        return np.concatenate((np.array([y_position, height, charw], dtype=np.float32), char_features))


def estimate_baseline_yposition(text_line):
    char_locations = ip.project(text_line, 0)
    yslice_stop_list = []
    for yslice, _ in char_locations:
        yslice_stop_list.append(yslice.stop)
    return round(np.median(yslice_stop_list))


def estimate_meanline_yposition(text_line):
    char_locations = ip.project(text_line, 0)
    yslice_start_list = []
    for yslice, _ in char_locations:
        yslice_start_list.append(yslice.start)
    return round(np.median(yslice_start_list))


class TemplateMatching(CharacterClassifier):

    def __init__(self, X, y):
        self.templates = []
        for i in range(len(X)):
            self.templates.append((X[i], y[i]))

    def classify(self, x, return_prob=False):
        scores = [(self.match(template, x), cls)
                  for template, cls in self.templates]
        nmi, cls = max(scores)
        if return_prob:
            return cls, 0.5 * nmi + 0.5
        else:
            return cls

    def match(self, template, character):
        max_nmi = -1
        for dy in range(-1, 5):
            for dx in range(-1, 5):
                nmi = normalized_match_index(template, character, dy, dx)
                max_nmi = max(nmi, max_nmi)
        return max_nmi


@njit
def normalized_match_index(template, object, dy, dx):
    oh, ow = object.shape
    th, tw = template.shape
    m = 0
    nm = 0
    for y in range(max(oh, th)):
        for x in range(max(ow, tw)):
            if 0 <= y + dy < oh and 0 <= x + dx < ow:
                op = object[y + dy, x + dx]
            else:
                op = BGROUND

            if y < th and x < tw:
                tp = template[y, x]
            else:
                tp = BGROUND

            if op != tp:
                nm += 1
            elif op != BGROUND:
                m += 1

    return (m-nm)/(m+nm)


class SklearnClassifier(CharacterClassifier):

    def __init__(self, classifier):
        self.classifier = classifier

    def classify(self, x, return_prob=False):
        ps = self.classifier.predict_proba([x])[0]
        cls = self.classifier.classes_[ps.argmax()]
        if return_prob:
            prob = ps.max()
            return cls, prob
        else:
            return cls


class SegmentationWithCharRecognition(TextLineRecognizer):

    def __init__(self, text_line_char_feature_extractor, char_classifier):
        self.feature_extractor = text_line_char_feature_extractor
        self.classify = char_classifier.classify

    def locate_characters(self, text_line):
        # return char locations
        raise NotImplementedError

    def recognize_text_line(self, text_line):
        char_locations = self.locate_characters(text_line)
        space_between_chars = [c2[1].start - c1[1].stop for c1,
                               c2 in zip(char_locations, char_locations[1:])]
        min_space_size = max(space_between_chars) / 2

        self.feature_extractor.set_text_line(text_line)

        char_loc = char_locations[0]
        char = text_line[char_loc]
        features = self.feature_extractor.extract_features(char, char_loc)
        recognized_text = self.classify(features)
        for char_loc, space_left_of_char in zip(char_locations[1:], space_between_chars):
            if space_left_of_char >= min_space_size:
                recognized_text += " "

            features = self.feature_extractor.extract_features(
                text_line[char_loc], char_loc)
            recognized_text += self.classify(features)

        return recognized_text


class VProjectionSegmentationWithCharRecognition(SegmentationWithCharRecognition):

    def __init__(self, text_line_char_feature_extractor, char_classifier, vprojection_threshold=0):
        super().__init__(text_line_char_feature_extractor, char_classifier)
        self.vprojection_threshold = vprojection_threshold

    def locate_characters(self, text_line):
        return ip.project(
            text_line, 0, self.vprojection_threshold)


class ConcompSegmentationWithCharRecognition(SegmentationWithCharRecognition):

    def locate_characters(self, text_line):
        return locate_text_line_concomp_chars(text_line)


def locate_text_line_concomp_chars(text_line):
    labels_im, num_labels = ndi.label(text_line)

    cc_labels = list(range(1, num_labels+1))
    cc_xslices = []
    cc_yslices = []
    for label in cc_labels:
        y_indices, x_indices = np.where(labels_im == label)
        cc_xslices.append(slice(x_indices.min(), x_indices.max()+1))
        cc_yslices.append(slice(y_indices.min(), y_indices.max()+1))

    cc_labels_to_connect_list = []
    passed_labels = set()
    for ii in range(len(cc_xslices)):
        if cc_labels[ii] in passed_labels:
            continue
        xslice1 = cc_xslices[ii]
        # yslice1 = cc_yslices[ii]
        center1 = (xslice1.start+xslice1.stop-1)/2
        cc_labels_to_connect_list.append([cc_labels[ii]])
        passed_labels.add(cc_labels[ii])
        for jj in range(ii+1, len(cc_xslices)):
            if cc_labels[jj] in passed_labels:
                continue
            xslice2 = cc_xslices[jj]
            # yslice2 = cc_yslices[jj]
            center2 = (xslice2.start+xslice2.stop-1)/2
            if xslice1.start <= center2 < xslice1.stop or xslice2.start <= center1 < xslice2.stop:
                cc_labels_to_connect_list[-1].append(cc_labels[jj])
                passed_labels.add(cc_labels[jj])

    char_slices = []
    for cc_labels_to_connect in cc_labels_to_connect_list:
        y_indices = None
        x_indices = None
        for label in cc_labels_to_connect:
            ys, xs = np.where(labels_im == label)
            y_indices = np.concatenate(
                (y_indices, ys)) if y_indices is not None else ys
            x_indices = np.concatenate(
                (x_indices, xs)) if x_indices is not None else xs

        cc_yslice = slice(y_indices.min(), y_indices.max()+1)
        cc_xslice = slice(x_indices.min(), x_indices.max()+1)
        char_slices.append((cc_yslice, cc_xslice))

    char_slices.sort(key=lambda x: x[1].start)

    return char_slices


class CharRecognitionBasedSegmentation(TextLineRecognizer):

    def __init__(self, text_line_char_feature_extractor, char_classifier, cuts_apart_number=3):
        self.feature_extractor = text_line_char_feature_extractor
        self.classify = char_classifier.classify
        self.cuts_apart_number = cuts_apart_number

    def recognize_text_line(self, text_line):
        self.feature_extractor.set_text_line(text_line)

        char_locations = ip.project(text_line, 0)
        space_between_chars = [c2[1].start - c1[1].stop for c1,
                               c2 in zip(char_locations, char_locations[1:])]
        min_space_size = max(space_between_chars) / 2

        word_locations = []
        word_loc = char_locations[0]
        for char_loc, space_left_of_char in zip(char_locations[1:], space_between_chars):
            if space_left_of_char >= min_space_size:
                word_locations.append(word_loc)
                word_loc = char_loc
            else:
                word_loc = sl.union(word_loc, char_loc)
        word_locations.append(word_loc)

        recognized_text = ""
        for idx, word_loc in enumerate(word_locations):
            word = text_line[word_loc]
            box, word = ip.bbox(word)
            word_loc = sl.shift(box, sl.start(word_loc))
            cuts = self.find_segmentation_cuts(word)
            recognized_word, _, _ = self.segment(word, word_loc, cuts)
            if idx > 0:
                recognized_text += " "
            recognized_text += recognized_word

        return recognized_text

    def find_segmentation_cuts(self, word):
        raise NotImplementedError

    def segment_char(self, word, cut1, cut2):
        # return char and it's location in word image
        raise NotImplementedError

    def clear_cache(self):
        self._cache = {}

    def _gen_key(self, cut1, cut2):
        return id(cut1), id(cut2)

    def cache(self, cut1, cut2, char, cost):
        key = self._gen_key(cut1, cut2)
        self._cache[key] = char, cost

    def in_cache(self, cut1, cut2):
        key = self._gen_key(cut1, cut2)
        return key in self._cache

    def get_cache(self, cut1, cut2):
        key = self._gen_key(cut1, cut2)
        return self._cache[key]

    def segment(self, word, word_loc, cuts, depth=0):
        if depth == 0:
            self.clear_cache()

        if len(cuts) < 2:
            return '', 0, []

        if len(cuts) == 2:

            if self.in_cache(cuts[0], cuts[1]):
                char, cost = self.get_cache(cuts[0], cuts[1])
            else:
                char, cost = self.recognize(word, word_loc, cuts[0], cuts[1])
                self.cache(cuts[0], cuts[1], char, cost)

            return char, cost, [(cuts[0], cuts[1])]

        chars_list = []
        costs = []
        segments_list = []
        for ii in range(1, min(self.cuts_apart_number+1, len(cuts))):

            if self.in_cache(cuts[0], cuts[ii]):
                char, cost = self.get_cache(cuts[0], cuts[ii])
            else:
                char, cost = self.recognize(word, word_loc, cuts[0], cuts[ii])
                self.cache(cuts[0], cuts[ii], char, cost)

            chars_list.append(char)
            costs.append(cost)
            segments_list.append([(cuts[0], cuts[ii])])

        asc_cost_indices = np.argsort(costs)
        for ii, jj in zip(asc_cost_indices, asc_cost_indices[1:]):
            chars, cost, segments = self.segment(
                word, word_loc, cuts[ii + 1:], depth+1)
            chars_list[ii] += chars
            costs[ii] += cost
            segments_list[ii] += segments
            if costs[ii] <= costs[jj]:
                break
        else:
            ii = asc_cost_indices[-1]
            chars, cost, segments = self.segment(
                word, word_loc, cuts[ii + 1:], depth+1)
            chars_list[ii] += chars
            costs[ii] += cost
            segments_list[ii] += segments

        desc_cost_indices = np.flip(np.argsort(costs))
        return chars_list[desc_cost_indices[-1]], costs[desc_cost_indices[-1]], segments_list[desc_cost_indices[-1]]

    def recognize(self, word, word_loc, cut1, cut2):
        char, loc = self.segment_char(word, cut1, cut2)
        # print(loc, sl.start(word_loc), sl.shift(loc, sl.start(word_loc)))
        features = self.feature_extractor.extract_features(
            char, sl.shift(loc, sl.start(word_loc)))

        char, prob = self.classify(features, return_prob=True)
        cost = -np.log(max(10**-10, prob))

        return char, cost


class CPSCCharacterRecognitionBasedSegmentation(CharRecognitionBasedSegmentation):

    def find_segmentation_cuts(self, word):
        return curved_pre_stroke_cuts(word)

    def segment_char(self, word, cut1, cut2):
        xslices = [slice(min(x1, x2), max(x1, x2))
                   for x1, x2 in zip(cut1, cut2)]
        mask = np.zeros_like(word)
        for y, xslice in enumerate(xslices):
            mask[y, xslice] = 1
        word_fground_and_mask = np.logical_and(word == FGROUND, mask)
        char = np.where(word_fground_and_mask, FGROUND,
                        BGROUND).astype(bool)

        return char, (slice(0, word.shape[0]), slice(0, word.shape[1]))


class BreakCostCharacterRecognitionBasedSegmentation(CharRecognitionBasedSegmentation):

    def __init__(self, text_line_char_feature_extractor, char_classifier, cuts_apart_number=3, threshold=1):
        super().__init__(text_line_char_feature_extractor, char_classifier, cuts_apart_number)
        self.threshold = threshold

    def find_segmentation_cuts(self, word):
        char_locations = locate_text_line_concomp_chars(word)
        cut_positions = []
        for idx, loc in enumerate(char_locations):
            if idx == 0:
                cut_positions.append(loc[1].start)

            char = word[loc]

            break_costs = []

            def fground_pixel_has_fground_neighbor_on_left(iline, oline):
                np.logical_and(iline[:-1] == FGROUND,
                               iline[1:] == FGROUND, out=oline)

            break_costs = ndi.generic_filter1d(
                char, fground_pixel_has_fground_neighbor_on_left, 2, mode='constant', cval=0).sum(axis=0)

            for i in range(1, len(break_costs)-1):
                if (1 <= break_costs[i] <= self.threshold):
                    cut_positions.append(i)

            cut_positions.append(loc[1].stop)

        return cut_positions

    def segment_char(self, word, cut1, cut2):
        loc = slice(0, word.shape[0]), slice(cut1, cut2)
        return word[loc], loc


# @njit
def curved_pre_stroke_cuts(image):
    """
    Implementation of 'Curved Pre-Stroke Cuts' from
    Segmentation of handprinted letter strings using a dynamic programming algorithm (https://ieeexplore.ieee.org/document/953902)
    """

    h, w = image.shape
    yc = h//2

    cost = np.full((h, w), np.finfo(np.float32).max, dtype=np.float32)
    source = np.full((h, w), -1, dtype=np.int32)
    yc_cost = np.full(w, np.finfo(np.float32).max, dtype=np.float32)
    yc_source = np.full(w, -1, dtype=np.int32)

    queue = []
    for i in range(w):
        queue.append((-1, i))

    while len(queue) > 0:
        y, x = queue.pop()
        if y == yc:
            continue

        if y == -1:
            prev_cost = 0.0
        else:
            prev_cost = cost[y, x]
        for dx in (-1, 0, 1):
            if not (0 <= x+dx < w):
                continue

            new_cost = prev_cost + 1.0
            if dx != 0:
                new_cost += 0.5
            new_point = y+1, x+dx
            if image[new_point] == FGROUND:
                new_cost += 2.0
            if new_cost < cost[new_point]:
                cost[new_point] = new_cost
                source[new_point] = x
                queue.append(new_point)

    queue = []
    for i in range(w):
        queue.append((h, i))

    while len(queue) > 0:
        y, x = queue.pop()
        if y == yc:
            continue

        if y == h:
            prev_cost = 0.0
        else:
            prev_cost = cost[y, x]
        for dx in (-1, 0, 1):
            if not (0 <= x+dx < w):
                continue

            new_cost = prev_cost + 1.0
            if dx != 0:
                new_cost += 0.5
            new_point = y-1, x+dx
            if image[new_point] == FGROUND:
                new_cost += 2.0
            curr_cost = yc_cost[new_point[1]
                                ] if new_point[0] == yc else cost[new_point]
            if new_cost < curr_cost:
                if new_point[0] == yc:
                    yc_cost[new_point[1]] = new_cost
                    yc_source[new_point[1]] = x
                else:
                    cost[new_point] = new_cost
                    source[new_point] = x
                queue.append(new_point)

    for i in range(w):
        cost[yc][i] += yc_cost[i]

    cuts = []
    for ii in range(1, w-1):
        if not (cost[yc][ii-1] >= cost[yc][ii] < cost[yc][ii+1]):
            continue

        cut = []
        py, px = yc, ii
        while py != -1:
            cut.insert(0, px)
            py, px = py-1, source[py, px]
        py, px = yc+1, yc_source[ii]
        while py != h:
            cut.append(px)
            py, px = py+1, source[py, px]

        cuts.append(cut)

    h, w = image.shape
    cuts.insert(0, [0]*h)
    cuts.append([w]*h)

    return cuts.copy()
