from util import levenshtein
from ocr import SimpleScannedPageOCR
from matplotlib.pyplot import imread
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
import data
import preprocessing
import layout_analysis
import text_line_recognition


lines_gtruth = data.load_text_lines_ground_truth(
    page_numbers=list(range(1, 7)))


feature_extractor = text_line_recognition.CharFeatureExtractor(lambda x: resize(x, (32,32)).flatten())

X = []
y = []
for (line_image, char_locations, char_classes), _ in lines_gtruth[:]:
    feature_extractor.set_text_line(line_image)
    for loc, cls in zip(char_locations, char_classes):
        X.append(feature_extractor.extract_features(line_image[loc], loc))
        y.append(cls)

line_segmenter = layout_analysis.RunLengthSmearing()
knn = KNeighborsClassifier(n_neighbors=1).fit(X,y)
char_classifier = text_line_recognition.SklearnClassifier(knn)
line_recognizer = text_line_recognition.ConcompSegmentationWithCharRecognition(feature_extractor, char_classifier)
ocr = SimpleScannedPageOCR(line_segmenter, line_recognizer)

page = imread('book/7.png')
recognized_text = ocr.recognize_text(page)

with open('book/7.trx', encoding='utf8') as f:
    gtruth_text = f.read()

print(gtruth_text)
print()
print(recognized_text)
print()

print("levenshtein", levenshtein(gtruth_text, recognized_text))