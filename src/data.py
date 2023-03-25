import os
import matplotlib.pyplot as plt
import sl
import preprocessing


def load_text_lines_ground_truth(book_path='book', page_numbers=None):
    book_pages_gtruth = {}
    for name in os.listdir(book_path):
        page_path = os.path.join(book_path, name)
        if not (page_path.endswith('.trx') or os.path.isdir(page_path)):
            continue

        page_number = int(os.path.splitext(name)[0])
        if page_numbers is not None and not page_number in page_numbers:
            continue

        page_gtruth = book_pages_gtruth.get(page_number, {})
        book_pages_gtruth[page_number] = page_gtruth

        if page_path.endswith('.trx'):
            with open(page_path, encoding='utf-8') as f:
                lines_transcript = [l[:-1] for l in f.readlines()]
                page_gtruth['lines_transcript'] = lines_transcript

        elif os.path.isdir(page_path):
            page_lines_gtruth = {}
            for name in os.listdir(page_path):
                line_path = os.path.join(page_path, name)
                if not (line_path.endswith('.png') or line_path.endswith('.txt') or os.path.isdir(line_path)):
                    continue

                line_number = int(os.path.splitext(name)[0])
                line_gtruth = page_lines_gtruth.get(line_number, {})
                page_lines_gtruth[line_number] = line_gtruth
                if line_path.endswith('.png'):
                    line_image = preprocessing.binarize(
                        plt.imread(line_path))
                    line_gtruth['line_image'] = line_image
                elif line_path.endswith('.txt'):
                    with open(line_path, encoding='utf-8') as f:
                        char_locations = []
                        for line in f.readlines():
                            raster = list(map(int, line.split(" ")))
                            char_locations.append(sl.box(*raster))
                        line_gtruth['char_locations'] = char_locations
                elif os.path.isdir(line_path):
                    with open(os.path.join(line_path, 'labels.txt'), encoding='utf-8') as f:
                        char_classes = []
                        for line in f.readlines():
                            _, cls = line[:-1].split(" ")
                            char_classes.append(cls)
                        line_gtruth['char_classes'] = char_classes

            line_numbers = list(page_lines_gtruth.keys())
            line_numbers.sort()
            lines_gtruth = []
            for line_number in line_numbers:
                line_gtruth = page_lines_gtruth[line_number]
                line_image = line_gtruth['line_image']
                char_locations = line_gtruth['char_locations']
                char_classes = line_gtruth['char_classes']
                lines_gtruth.append((line_image, char_locations, char_classes))
            page_gtruth['lines_gtruth'] = lines_gtruth

    page_numbers = list(book_pages_gtruth.keys())
    page_numbers.sort()

    pages_gtruth = []
    for page_number in page_numbers:
        page_gtruth = book_pages_gtruth[page_number]
        lines_gtruth = page_gtruth['lines_gtruth']
        lines_transcript = page_gtruth['lines_transcript']
        for line_gtruth, line_transcript in zip(lines_gtruth, lines_transcript):
            pages_gtruth.append((line_gtruth, line_transcript))

    return pages_gtruth
