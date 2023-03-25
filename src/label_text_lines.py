
# Implementation of move and zoom of tkinter canvas with mouse taken from:
# https://stackoverflow.com/questions/25787523/move-and-zoom-a-tkinter-canvas-with-mouse/48069295#48069295

import os
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

import sl
import image_processing
import preprocessing
import layout_analysis


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

    def __init__(self, mainframe, path):
        ''' Initialize the main Frame '''
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Text lines labeler')
        # Vertical and horizontal scrollbars for canvas
        vbar = AutoScrollbar(self.master, orient='vertical')
        hbar = AutoScrollbar(self.master, orient='horizontal')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='we')
        # Open image
        self.image = Image.open(path)
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

        def mousexy_to_canvasxy(x, y):
            x = self.canvas.canvasx(x)
            y = self.canvas.canvasy(y)
            return x, y

        def canvasxy_to_imagexy(x, y):
            dx, dy = self.canvas.coords(self.text)
            s = self.imscale
            x = round((x - dx) / s)
            y = round((y - dy) / s)
            return x, y

        def mousexy_to_imagexy(x, y):
            x, y = mousexy_to_canvasxy(x, y)
            return canvasxy_to_imagexy(x, y)

        def imagexy_to_canvasxy(x, y):
            dx, dy = self.canvas.coords(self.text)
            s = self.imscale
            x = round(x * s + dx)
            y = round(y * s + dy)
            return x, y

        def rectangle(vertex0, vertex1):
            # rectangle is specified as two points: (x0, y0) is the top left corner,
            # and (x1, y1) is the location of the pixel just outside of the bottom right corner
            x0, y0 = vertex0
            x1, y1 = vertex1
            return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

        def draw_rectangle(rect, outline='blue'):
            x0, y0, x1, y1 = rect
            x0, y0 = imagexy_to_canvasxy(x0, y0)
            x1, y1 = imagexy_to_canvasxy(x1, y1)
            return self.canvas.create_rectangle(x0, y0, x1, y1, outline=outline)

        page = plt.imread(path)
        page = preprocessing.binarize(page)

        self.rect_id_pairs = []

        img_fname = self.image.filename
        base_fname = img_fname[:img_fname.rindex('.')]
        txt_fname = os.path.join(base_fname + '.txt')
        if os.path.isfile(txt_fname):
            with open(base_fname + '.txt', 'r') as f:
                for line in f.readlines():
                    rect = list(map(int, line.split(" ")))
                    id = draw_rectangle(rect)
                    self.rect_id_pairs.append((rect, id))
        else:
            locate_text_lines = layout_analysis.RunLengthSmearing().locate_text_lines
            text_line_locations = locate_text_lines(page)
            for loc in text_line_locations:
                # loc = sl.pad(loc, 4)
                rect = sl.raster(loc)
                id = draw_rectangle(rect)
                self.rect_id_pairs.append((rect, id))

        self.mouse_pressed = False
        self.rectid = None

        def save_xy_as_rect_vertex(event):
            self.vertex = mousexy_to_imagexy(event.x, event.y)
            self.mouse_pressed = True

        def use_xy_as_rects_opposite_vertex(event):
            if not self.mouse_pressed:
                return

            if self.rectid:
                self.canvas.delete(self.rectid)

            opposite_vertex = mousexy_to_imagexy(event.x, event.y)
            rect = rectangle(self.vertex, opposite_vertex)
            self.rectid = draw_rectangle(rect, outline='red')

        def add_merge_or_remove_min_rect(event):
            if not self.mouse_pressed:
                return

            self.mouse_pressed = False

            if self.rectid:
                self.canvas.delete(self.rectid)
                self.rectid = None

            opposite_vertex = mousexy_to_imagexy(event.x, event.y)
            x0, y0, x1, y1 = rectangle(self.vertex, opposite_vertex)

            selected = []
            for idx, (rect, _) in enumerate(self.rect_id_pairs):
                if x0 <= rect[0] and x1 >= rect[2] and y0 <= rect[1] and y1 >= rect[3]:
                    selected.append(idx)

            if len(selected) == 1:
                self.canvas.delete(self.rect_id_pairs.pop(selected[0])[1])

            elif len(selected) > 1:
                x0 = min([self.rect_id_pairs[idx][0][0] for idx in selected])
                y0 = min([self.rect_id_pairs[idx][0][1] for idx in selected])
                x1 = max([self.rect_id_pairs[idx][0][2] for idx in selected])
                y1 = max([self.rect_id_pairs[idx][0][3] for idx in selected])

                for idx in reversed(selected):
                    self.canvas.delete(self.rect_id_pairs.pop(idx)[1])

                rect = (x0, y0, x1, y1)
                id = draw_rectangle(rect)
                self.rect_id_pairs.append((rect, id))

            else:
                h, w = page.shape
                region_yslice = slice(max(y0, 0), min(y1, h))
                region_xslice = slice(max(x0, 0), min(x1, w))
                region_slice = region_yslice, region_xslice
                page_region = page[region_slice]
                text_line_locations = image_processing.project(page_region, 1)
                for loc in text_line_locations:
                    # x0, x1, y0, y1 = sl.raster(loc)
                    # if x1-x0 < 10 or y1-y0 < 10:
                    #     continue

                    rect = sl.raster(sl.shift(loc, sl.start(region_slice)))
                    id = draw_rectangle(rect)
                    self.rect_id_pairs.append((rect, id))

        def remove_rectangle(_):
            if self.rectid:
                self.canvas.delete(self.rectid)
                self.rectid = None

        def enable_correcting_rect_side_if_near(event):
            x, y = mousexy_to_imagexy(event.x, event.y)

            for idx, (rect, _) in enumerate(self.rect_id_pairs):

                corner_margin = 10

                if rect[0]-corner_margin <= x <= rect[0]+corner_margin and rect[1]-corner_margin < y < rect[1]+corner_margin:
                    self.canvas.config(cursor="top_left_corner")
                    self.rect_idx = idx
                    self.fixed_corner = (rect[2], rect[3])
                    self.canvas.bind('<B1-Motion>', correct_rectangle_corner)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break
                if rect[0]-corner_margin <= x <= rect[0]+corner_margin and rect[3]-corner_margin < y < rect[3]+corner_margin:
                    self.canvas.config(cursor="bottom_left_corner")
                    self.rect_idx = idx
                    self.fixed_corner = (rect[2], rect[1])
                    self.canvas.bind('<B1-Motion>', correct_rectangle_corner)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break
                if rect[2]-corner_margin <= x <= rect[2]+corner_margin and rect[1]-corner_margin < y < rect[1]+corner_margin:
                    self.canvas.config(cursor="top_right_corner")
                    self.rect_idx = idx
                    self.fixed_corner = (rect[0], rect[3])
                    self.canvas.bind('<B1-Motion>', correct_rectangle_corner)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break
                if rect[2]-corner_margin <= x <= rect[2]+corner_margin and rect[3]-corner_margin < y < rect[3]+corner_margin:
                    self.canvas.config(cursor="bottom_right_corner")
                    self.rect_idx = idx
                    self.fixed_corner = (rect[0], rect[1])
                    self.canvas.bind('<B1-Motion>', correct_rectangle_corner)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break

                side_margin = 5

                if rect[0]-side_margin <= x <= rect[0]+side_margin and rect[1] < y < rect[3]:
                    self.canvas.config(cursor="left_side")
                    self.rect_idx = idx
                    self.side_idx = 0
                    self.canvas.bind('<B1-Motion>', correct_rectangle_side)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break
                if rect[2]-side_margin <= x <= rect[2]+side_margin and rect[1] < y < rect[3]:
                    self.canvas.config(cursor="right_side")
                    self.rect_idx = idx
                    self.side_idx = 2
                    self.canvas.bind('<B1-Motion>', correct_rectangle_side)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break
                if rect[1]-side_margin <= y <= rect[1]+side_margin and rect[0] < x < rect[2]:
                    self.canvas.config(cursor="top_side")
                    self.rect_idx = idx
                    self.side_idx = 1
                    self.canvas.bind('<B1-Motion>', correct_rectangle_side)
                    self.canvas.bind("<ButtonRelease-1>",
                                     unmark_corrected_rect)
                    break
                if rect[3]-side_margin <= y <= rect[3]+side_margin and rect[0] < x < rect[2]:
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
            self.rect_id_pairs.sort(key=lambda x: x[0][1])

            img_name = self.image.filename

            base_fname = img_name[:img_name.rindex('.')]
            with open(base_fname + '.txt', 'w') as f:
                for (rect, _) in self.rect_id_pairs:
                    x0, y0, x1, y1 = rect
                    f.write(f'{x0} {y0} {x1} {y1}\n')

            dir_path = img_name[:img_name.rindex('.')]
            Path(dir_path).mkdir(parents=True, exist_ok=True)

            fname_suffix = img_name[img_name.rindex('.')+1:]
            for idx, (rect, _) in enumerate(self.rect_id_pairs):
                base_fname = str(idx)
                full_path_fname = os.path.join(
                    dir_path, base_fname + "." + fname_suffix)
                # plt.imsave(full_path_fname,
                #            page[sl.box(*rect)], cmap='gray_r')
                self.image.crop(rect).save(full_path_fname)

            self.master.destroy()

        def quit(_):
            self.master.destroy()

        self.canvas.bind("<Control-ButtonPress-1>", save_xy_as_rect_vertex)
        self.canvas.bind("<Control-Motion>", use_xy_as_rects_opposite_vertex)
        self.canvas.bind("<Control-ButtonRelease-1>",
                         add_merge_or_remove_min_rect)

        self.master.bind("<KeyRelease>", remove_rectangle)

        self.canvas.bind("<Motion>", enable_correcting_rect_side_if_near)

        self.master.bind("<Control-s>", save)
        self.master.bind("<Control-S>", save)

        self.master.bind("<Control-w>", quit)
        self.master.bind("<Control-w>", quit)

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


path = sys.argv[1]
root = tk.Tk()
app = Zoom(root, path)
root.mainloop()
