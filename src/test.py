from util import levenshtein
from features import gsc_features
from ocr import SimpleScannedPageOCR
from matplotlib.pyplot import imread
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
import sl
import preprocessing
import layout_analysis
import text_line_recognition
import image_processing as ip

pagec = imread('page/pagec.png')
pagec = preprocessing.binarize(pagec)
preprocessing.remove_salt_pepper(pagec)
pagec = preprocessing.correct_skew(pagec)

extract_char_features = lambda x: gsc_features(resize(x, (32,32))).flatten()

feature_extractor = text_line_recognition.CharFeatureExtractor(extract_char_features)

X = []
line_locs = ip.project(pagec, 1)
for line_loc in line_locs[:]:
    line = pagec[line_loc]
    feature_extractor.set_text_line(line)
    char_locs = ip.project(pagec[line_loc], 0)
    for loc in char_locs:
        loc = sl.shift(loc, sl.start(line_loc))
        char = pagec[loc]
        x = feature_extractor.extract_features(char, loc)
        X.append(x)

with open('page/pagec.txt') as f:
    y = list(f.read().replace('\n', ''))

line_segmenter = layout_analysis.HorizontalProjection(threshold=2)
knn = KNeighborsClassifier(n_neighbors=1).fit(X,y)
char_classifier = text_line_recognition.SklearnClassifier(knn)
line_recognizer = text_line_recognition.ConcompSegmentationWithCharRecognition(feature_extractor, char_classifier)
ocr = SimpleScannedPageOCR(line_segmenter, line_recognizer)

paged = imread('page/paged.png')
recognized_text = ocr.recognize_text(paged)

with open('page/paged.txt', encoding='utf8') as f:
    gtruth_text = f.read()

print(gtruth_text)
print()
print(recognized_text)
print()

print("levenshtein", levenshtein(gtruth_text, recognized_text))