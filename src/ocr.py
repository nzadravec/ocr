from skimage.filters import threshold_otsu
from preprocessing import binarize, correct_skew


class SimpleScannedPageOCR:
    """ page → text """

    def __init__(self, text_line_segmenter, text_line_recognizer, binarization_threshold=threshold_otsu):
        self.segment_text_lines = text_line_segmenter.segment_text_lines
        self.recognize_text_line = text_line_recognizer.recognize_text_line
        self.binarization_threshold = binarization_threshold

    def recognize_text(self, page):
        page = binarize(page, self.binarization_threshold)
        page = correct_skew(page)

        page_text = ""
        for line in self.segment_text_lines(page):
            line_text = self.recognize_text_line(line)
            page_text += line_text + '\n'

        return page_text
