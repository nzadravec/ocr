# Implementation of move and zoom of tkinter canvas with mouse taken from:
# https://stackoverflow.com/questions/25787523/move-and-zoom-a-tkinter-canvas-with-mouse/48069295#48069295

import os
import sys
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from PIL import Image, ImageTk
from numba import njit
import numpy as np
import matplotlib.pyplot as plt

from constants import FGROUND, BGROUND
import sl
import preprocessing
import text_line_recognition


class AutoScrollbar(ttk.Scrollbar):
    ''' A scrollbar that hides itself if it's not needed.
        Works only if you use the grid geometry manager '''

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
        ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with this widget')

    def place(self, **kw):
        raise tk.TclError('Cannot use place with this widget')


class Zoom(ttk.Frame):
    ''' Simple zoom with mouse wheel '''

    def __init__(self, mainframe, img_path, trx_path):
        ''' Initialize the main Frame '''
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Text line characters labeler')
        # Vertical and horizontal scrollbars for canvas
        vbar = AutoScrollbar(self.master, orient='vertical')
        hbar = AutoScrollbar(self.master, orient='horizontal')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='we')
        # Open image
        self.image = Image.open(img_path)
        # Create canvas and put image on it
        self.canvas = tk.Canvas(self.master, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        # bind scrollbars to the canvas
        vbar.configure(command=self.canvas.yview)
        hbar.configure(command=self.canvas.xview)
        # Make the canvas expandable
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        # Bind events to the Canvas
        self.canvas.bind('<ButtonPress-1>', self.move_from)
        self.canvas.bind('<B1-Motion>',     self.move_to)
        # with Windows and MacOS, but not Linux
        self.canvas.bind('<MouseWheel>', self.wheel)
        # only with Linux, wheel scroll down
        self.canvas.bind('<Button-5>',   self.wheel)
        # only with Linux, wheel scroll up
        self.canvas.bind('<Button-4>',   self.wheel)
        # Show image and plot some random test rectangles on the canvas
        self.imscale = 1.0
        self.imageid = None
        self.delta = 0.75
        # Text is used to set proper coordinates to the image. You can make it invisible.
        self.text = self.canvas.create_text(
            0, 0, anchor='nw', text='')
        self.show_image()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

        text_line_img = plt.imread(img_path)
        text_line_img = preprocessing.binarize(text_line_img)

        self.line_txt = {}
        if trx_path:
            with open(trx_path, 'r', encoding='utf8') as f:
                base_img_filename = img_path[:img_path.rindex('.')]
                line_num = base_img_filename[img_path.rindex(os.path.sep)+1:]
                self.line_txt = list(f.readlines())[
                    int(line_num)].replace(' ', '')
                if self.line_txt[-1] == '\n':
                    self.line_txt = self.line_txt[:-1]
                self.line_txt = dict(enumerate(self.line_txt))

        def canvasxy_to_imagexy(x, y):
            dx, dy = self.canvas.coords(self.text)
            s = self.imscale
            x = round((x - dx) / s)
            y = round((y - dy) / s)
            return x, y

        def imagexy_to_canvasxy(x, y):
            dx, dy = self.canvas.coords(self.text)
            s = self.imscale
            x = round(x * s + dx)
            y = round(y * s + dy)
            return x, y

        def draw_rectangle(rect, outline='blue', label=None):
            x0, y0 = imagexy_to_canvasxy(rect[0], rect[1])
            x1, y1 = imagexy_to_canvasxy(rect[2], rect[3])
            id = self.canvas.create_rectangle(x0, y0, x1, y1, outline=outline)
            if label:
                x0, y0 = imagexy_to_canvasxy(rect[0], 0)
                self.canvas.create_text(
                    x0, y0, anchor='sw', text=label, fill='red', font=(20))
            return id

        img_filename = self.image.filename
        base_filename = img_filename[:img_filename.rindex('.')]
        txt_filename = os.path.join(base_filename + '.txt')
        rects = []
        if os.path.isfile(txt_filename):
            with open(base_filename + '.txt', 'r') as f:
                for line in f.readlines():
                    rect = list(map(int, line.split(" ")))
                    rects.append(rect)
        else:
            # char_locations = image_processing.project(text_line_img, 0)
            char_locations = text_line_recognition.locate_text_line_concomp_chars(
                text_line_img)
            for loc in char_locations:
                # loc = sl.pad(loc, 1)
                rect = sl.raster(loc)
                rects.append(rect)

        self.rect_id_pairs = []
        for idx, rect in enumerate(rects):
            id = draw_rectangle(rect, label=self.line_txt.get(idx, ' '))
            self.rect_id_pairs.append((rect, id))

        def eventxy_to_canvasxy(x, y):
            x = self.canvas.canvasx(x)
            y = self.canvas.canvasy(y)
            return x, y

        def mousexy_to_imagexy(x, y):
            x, y = eventxy_to_canvasxy(x, y)
            return canvasxy_to_imagexy(x, y)

        self.mouse_button_pressed = False
        self.rectid = None

        def save_xy(event):
            x, y = eventxy_to_canvasxy(event.x, event.y)
            self.x, self.y = canvasxy_to_imagexy(x, y)
            self.mouse_button_pressed = True

        def box(x0, y0, x1, y1):
            # box is a 4-tuple defining the left, upper, right, and lower pixel
            return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

        def rectangle(vertex0, vertex1):
            # rectangle is a 4-tuple defining the left, upper, right, and lower pixel
            x0, y0 = vertex0
            x1, y1 = vertex1
            return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

        def draw_rect(event):
            if not self.mouse_button_pressed:
                return

            x, y = eventxy_to_canvasxy(event.x, event.y)
            x, y = canvasxy_to_imagexy(x, y)
            if self.rectid:
                self.canvas.delete(self.rectid)
            self.rectid = draw_rectangle(
                box(self.x, self.y, x, y), outline='red')

        def merge_bbs_in_rect(event):
            if not self.mouse_button_pressed:
                return

            self.mouse_button_pressed = False

            self.canvas.delete(self.rectid)
            self.rectid = None

            x, y = eventxy_to_canvasxy(event.x, event.y)
            x, y = canvasxy_to_imagexy(x, y)

            x0, y0, x1, y1 = box(self.x, self.y, x, y)

            selected = []
            for idx, (rect, _) in enumerate(self.rect_id_pairs):
                if x0 <= rect[0] and x1 >= rect[2] and y0 <= rect[1] and y1 >= rect[3]:
                    selected.append(idx)

            if len(selected) == 0:
                h, w = text_line_img.shape
                rect_yslice = slice(max(y0, 0), min(y1, h))
                rect_xslice = slice(max(x0, 0), min(x1, w))
                rect_slice = rect_yslice, rect_xslice
                line_img_part = text_line_img[rect_slice]
                # char_locations = image_processing.project(line_img_part, 0)
                char_locations = text_line_recognition.locate_text_line_concomp_chars(
                    line_img_part)
                for loc in char_locations:
                    rect = sl.raster(sl.shift(loc, sl.start(rect_slice)))
                    id = draw_rectangle(rect)
                    self.rect_id_pairs.append((rect, id))
            elif len(selected) == 1:
                self.canvas.delete(self.rect_id_pairs.pop(selected[0])[1])
            else:
                x0 = min([self.rect_id_pairs[idx][0][0] for idx in selected])
                y0 = min([self.rect_id_pairs[idx][0][1] for idx in selected])
                x1 = max([self.rect_id_pairs[idx][0][2] for idx in selected])
                y1 = max([self.rect_id_pairs[idx][0][3] for idx in selected])

                for idx in reversed(selected):
                    self.canvas.delete(self.rect_id_pairs.pop(idx)[1])

                rect = (x0, y0, x1, y1)
                id = draw_rectangle(rect)
                self.rect_id_pairs.append((rect, id))

            refresh(None)

        self.vlineid = None

        def draw_vline(event):
            x, _ = eventxy_to_canvasxy(event.x, 0)
            _, height = self.image.size
            _, y0 = imagexy_to_canvasxy(0, 0)
            _, y1 = imagexy_to_canvasxy(0, height)

            if self.vlineid:
                self.canvas.delete(self.vlineid)
            self.vlineid = self.canvas.create_line(x, y0, x, y1, fill='red')

        def remove_rect_and_vline(_):
            if self.rectid:
                self.canvas.delete(self.rectid)
                self.rectid = None
            if self.vlineid:
                self.canvas.delete(self.vlineid)
                self.vlineid = None

        @njit
        def min_bounding_box(glyph):
            h, w = glyph.shape
            min_bb = np.empty(4, dtype=np.int32)
            fground_pixel_found = False
            for y in range(h):
                for x in range(w):
                    if glyph[y, x] == FGROUND:
                        min_bb[0] = y
                        fground_pixel_found = True
                        break
                if fground_pixel_found:
                    break
            fground_pixel_found = False
            for y in range(h-1, -1, -1):
                for x in range(w):
                    if glyph[y, x] == FGROUND:
                        min_bb[1] = y+1
                        fground_pixel_found = True
                        break
                if fground_pixel_found:
                    break
            fground_pixel_found = False
            for x in range(w):
                for y in range(h):
                    if glyph[y, x] == FGROUND:
                        min_bb[2] = x
                        fground_pixel_found = True
                        break
                if fground_pixel_found:
                    break
            fground_pixel_found = False
            for x in range(w-1, -1, -1):
                for y in range(h):
                    if glyph[y, x] == FGROUND:
                        min_bb[3] = x+1
                        fground_pixel_found = True
                        break
                if fground_pixel_found:
                    break
            return min_bb

        def split_rect_with_vline(event):
            x, _ = eventxy_to_canvasxy(event.x, 0)
            x, _ = canvasxy_to_imagexy(x, 0)

            selected = None
            for idx, (rect, _) in enumerate(self.rect_id_pairs):
                if rect[0] <= x < rect[2]:
                    selected = idx
                    break

            if selected is None:
                return

            rect, id = self.rect_id_pairs.pop(selected)
            self.canvas.delete(id)

            x0, y0, x1, y1 = rect
            line_img_region = text_line_img[y0:y1, x0:x]
            y0, y1, x0, x1 = min_bounding_box(line_img_region)
            rect0 = (rect[0] + x0, rect[1] + y0,
                     rect[0] + x1, rect[1] + y1)

            x0, y0, x1, y1 = rect
            line_img_region = text_line_img[y0:y1, x:x1]
            y0, y1, x0, x1 = min_bounding_box(line_img_region)
            rect1 = (x + x0, rect[1] + y0, x + x1, rect[1] + y1)

            id0 = draw_rectangle(rect0)
            self.rect_id_pairs.append((rect0, id0))
            id1 = draw_rectangle(rect1)
            self.rect_id_pairs.append((rect1, id1))

            refresh(None)

        def enable_correcting_rect_side_if_near(event):
            x, y = mousexy_to_imagexy(event.x, event.y)

            for idx, (rect, _) in enumerate(self.rect_id_pairs):
                if rect[0]-1 <= x <= rect[0]+1 and rect[1]-1 < y < rect[1]+1:
                    self.canvas.config(cursor="top_left_corner")
                    self.rect_idx = idx
                    self.fixed_corner = (rect[2], rect[3])
                    self.canvas.bind('<B1-Motion>', correct_rectangle_corner)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break
                if rect[0]-1 <= x <= rect[0]+1 and rect[3]-1 < y < rect[3]+1:
                    self.canvas.config(cursor="bottom_left_corner")
                    self.rect_idx = idx
                    self.fixed_corner = (rect[2], rect[1])
                    self.canvas.bind('<B1-Motion>', correct_rectangle_corner)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break
                if rect[2]-1 <= x <= rect[2]+1 and rect[1]-1 < y < rect[1]+1:
                    self.canvas.config(cursor="top_right_corner")
                    self.rect_idx = idx
                    self.fixed_corner = (rect[0], rect[3])
                    self.canvas.bind('<B1-Motion>', correct_rectangle_corner)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break
                if rect[2]-1 <= x <= rect[2]+1 and rect[3]-1 < y < rect[3]+1:
                    self.canvas.config(cursor="bottom_right_corner")
                    self.rect_idx = idx
                    self.fixed_corner = (rect[0], rect[1])
                    self.canvas.bind('<B1-Motion>', correct_rectangle_corner)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break

                if rect[0] <= x <= rect[0] and rect[1] < y < rect[3]:
                    self.canvas.config(cursor="left_side")
                    self.rect_idx = idx
                    self.side_idx = 0
                    self.canvas.bind('<B1-Motion>', correct_rectangle_side)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break
                if rect[2] <= x <= rect[2] and rect[1] < y < rect[3]:
                    self.canvas.config(cursor="right_side")
                    self.rect_idx = idx
                    self.side_idx = 2
                    self.canvas.bind('<B1-Motion>', correct_rectangle_side)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break
                if rect[1] <= y <= rect[1] and rect[0] < x < rect[2]:
                    self.canvas.config(cursor="top_side")
                    self.rect_idx = idx
                    self.side_idx = 1
                    self.canvas.bind('<B1-Motion>', correct_rectangle_side)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break
                if rect[3] <= y <= rect[3] and rect[0] < x < rect[2]:
                    self.canvas.config(cursor="bottom_side")
                    self.rect_idx = idx
                    self.side_idx = 3
                    self.canvas.bind('<B1-Motion>', correct_rectangle_side)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break
            else:
                self.canvas.config(cursor="")
                self.canvas.bind('<B1-Motion>', self.move_to)
                self.canvas.bind("<ButtonRelease-1>", None)

        def correct_rectangle_corner(event):
            _, id = self.rect_id_pairs[self.rect_idx]
            self.canvas.delete(id)

            new_corner = mousexy_to_imagexy(event.x, event.y)

            newrect = rectangle(self.fixed_corner, new_corner)
            newid = draw_rectangle(newrect, outline='red')
            self.rect_id_pairs[self.rect_idx] = (newrect, newid)

        def correct_rectangle_side(event):
            rect, id = self.rect_id_pairs[self.rect_idx]
            self.canvas.delete(id)

            x, y = mousexy_to_imagexy(event.x, event.y)

            rect = list(rect)
            rect[self.side_idx] = x if self.side_idx % 2 == 0 else y
            newrect = rectangle(rect[0:2], rect[2:4])
            newid = draw_rectangle(newrect, outline='red')
            self.rect_id_pairs[self.rect_idx] = (newrect, newid)

        def unmark_corrected_rect(_):
            rect, id = self.rect_id_pairs[self.rect_idx]
            self.canvas.delete(id)
            newid = draw_rectangle(rect)
            self.rect_id_pairs[self.rect_idx] = (rect, newid)

        def save(_):
            self.rect_id_pairs.sort(key=lambda x: x[0][0])

            img_filename = self.image.filename
            base_filename = img_filename[:img_filename.rindex('.')]
            with open(base_filename + '.txt', 'w') as f:
                for idx, (rect, _) in enumerate(self.rect_id_pairs):
                    x0, y0, x1, y1 = rect
                    f.write(f'{x0} {y0} {x1} {y1}\n')

            img_filename = self.image.filename
            dir_path = img_filename[:img_filename.rindex('.')]
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

            Path(dir_path).mkdir(parents=True, exist_ok=True)
            filename_suffix = img_filename[img_filename.rindex('.')+1:]

            with open(os.path.join(dir_path, 'labels.txt'), 'w', encoding='utf8') as f:
                for idx, (rect, _) in enumerate(self.rect_id_pairs):
                    base_filename = str(idx)
                    filename = base_filename + "." + filename_suffix
                    full_path_filename = os.path.join(dir_path, filename)
                    self.image.crop(rect).save(full_path_filename)

                    f.write(f"{filename} {self.line_txt.get(idx)}\n")

            self.master.destroy()

        def quit(_):
            self.master.destroy()

        def refresh(_):
            if _ is not None:
                if trx_path:
                    with open(trx_path, 'r', encoding='utf8') as f:
                        base_img_filename = img_path[:img_path.rindex('.')]
                        line_num = base_img_filename[img_path.rindex(
                            os.path.sep)+1:]
                        self.line_txt = list(f.readlines())[
                            int(line_num)].replace(' ', '')
                        if self.line_txt[-1] == '\n':
                            self.line_txt = self.line_txt[:-1]
                        self.line_txt = dict(enumerate(self.line_txt))

            for id in self.canvas.find_all():
                if id == self.text or id == self.imageid:
                    continue
                self.canvas.delete(id)

            self.rect_id_pairs.sort(key=lambda x: x[0][0])

            for idx in range(len(self.rect_id_pairs)):
                rect, _ = self.rect_id_pairs[idx]
                id = draw_rectangle(rect, label=self.line_txt.get(idx, ' '))
                self.rect_id_pairs[idx] = (rect, id)

        self.canvas.bind("<Control-ButtonPress-1>", save_xy)
        self.canvas.bind("<Control-Motion>", draw_rect)
        self.canvas.bind("<Control-ButtonRelease-1>", merge_bbs_in_rect)

        self.canvas.bind("<Shift-Motion>", draw_vline)
        self.canvas.bind("<Shift-Button-1>", split_rect_with_vline)
        self.master.bind("<KeyRelease>", remove_rect_and_vline)

        self.canvas.bind("<Motion>", enable_correcting_rect_side_if_near)

        self.master.bind("<Control-s>", save)
        self.master.bind("<Control-S>", save)

        self.master.bind("<Control-w>", quit)
        self.master.bind("<Control-W>", quit)

        self.master.bind("<Control-r>", refresh)
        self.master.bind("<Control-R>", refresh)

        self.master.protocol("WM_DELETE_WINDOW", lambda: save(None))

    def move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        if event.state == 0:
            self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        if event.state == 256:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def wheel(self, event):
        ''' Zoom with mouse wheel '''
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta < 0:
            scale *= self.delta
            self.imscale *= self.delta
        if event.num == 4 or event.delta > 0:
            scale /= self.delta
            self.imscale /= self.delta
        # Rescale all canvas objects
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.canvas.scale('all', x, y, scale, scale)
        self.show_image()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def show_image(self):
        ''' Show image on the Canvas '''
        if self.imageid:
            self.canvas.delete(self.imageid)
            self.imageid = None
            self.canvas.imagetk = None  # delete previous image from the canvas
        width, height = self.image.size
        new_size = int(self.imscale * width), int(self.imscale * height)
        imagetk = ImageTk.PhotoImage(self.image.resize(new_size))
        # Use self.text object to set proper coordinates
        self.imageid = self.canvas.create_image(self.canvas.coords(self.text),
                                                anchor='nw', image=imagetk)
        self.canvas.lower(self.imageid)  # set it into background
        # keep an extra reference to prevent garbage-collection
        self.canvas.imagetk = imagetk


img_path = sys.argv[1]
trx_path = sys.argv[2] if len(sys.argv) > 2 else None
root = tk.Tk()
app = Zoom(root, img_path, trx_path)
root.mainloop()
