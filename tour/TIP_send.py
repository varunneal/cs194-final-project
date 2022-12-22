from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


def find_line(x1, y1, x2, y2, c):
    # Given two points (x1, y1) and (x2, y2), find a fitting line L
    # Then, return x such that L(x) = c  and y such that L(c) = y
    m = (y1 - y2) / (x1 - x2)
    b = y1 - m * x1
    return (c - b) / m, m * c + b


class TIP:
    def __init__(self, im):
        """
        :param im: RGB image
        Stores
        - x and y coordinates of inner rectangle of image (irx, iry)
        - x and y coordinates of vanishing point (vx, vy)
        - x and y coordinates of outer polygon (orx, ory)
        """
        self.im = im
        self.iry = None
        self.irx = None
        self.vx = None
        self.vy = None
        self.orx = None
        self.ory = None


    def select_rectangle(self):
        plt.ion()
        txt = "Select rectangle points"
        plt.figtext(0.5, 0.1, txt, wrap=True, horizontalalignment='center', fontsize=12, c='r', backgroundcolor='k')

        plt.imshow(self.im)
        p0, p1 = plt.ginput(2)

        plt.ioff()
        plt.clf()

        self.irx = np.array([p0[0], p1[0], p1[0], p0[0], p0[0]], dtype=int)
        self.iry = np.array([p0[1], p0[1], p1[1], p1[1], p0[1]], dtype=int)
        self.vx = np.mean(self.irx)
        self.vy = np.mean(self.iry)

    def find_corner(self, idx):
        y_max, x_max, _ = self.im.shape
        x_bounds = [0, x_max, x_max, 0]
        y_bounds = [0, 0, y_max, y_max]
        boundary_x = x_bounds[idx]
        boundary_y = y_bounds[idx]

        y1 = boundary_y
        x1 = find_line(self.vx, self.vy, self.irx[idx], self.iry[idx], boundary_y)[0]
        x2 = boundary_x
        y2 = find_line(self.vx, self.vy, self.irx[idx], self.iry[idx], boundary_x)[1]
        if np.linalg.norm([self.vx - x1, self.vy - y1]) > np.linalg.norm([self.vx - x2, self.vy - y2]):
            return x1, y1
        else:
            return x2, y2

    def get_planar_lines(self):
        y_max, x_max, _ = self.im.shape
        orx, ory = [], []
        for i in range(4):
            x, y = self.find_corner(i)  #
            orx.append(int(x))
            ory.append(int(y))
        self.orx = np.array(orx)
        self.ory = np.array(ory)

        plt.plot(self.irx, self.iry, 'b')  # rectangle
        plt.plot(self.vx, self.vy, 'b')  # vanishing point
        for i in range(4):  # planar lines
            plt.plot([self.vx, self.irx[i]], [self.vy, self.iry[i]], 'r-.')
            plt.plot([self.orx[i], self.irx[i]], [self.ory[i], self.iry[i]], 'r')
            plt.draw()
        plt.imshow(self.im)
        plt.figtext(0.5, 0.05, "Select vanishing point. Finalize with \n ENTER. Fullscreen recommended.", wrap=True,
                    horizontalalignment='center', fontsize=10, c='r', backgroundcolor='k')

    def select_vanishing_point(self):
        """
        GUI for selecting vanishing point. Allows user to retry until satisfied.
        """
        plt.clf()
        self.get_planar_lines()
        plt.pause(0.1)

        def mouse_released(event):
            if event.button != 1:
                return
            plt.clf()
            self.vx, self.vy = event.xdata, event.ydata
            self.get_planar_lines()
            plt.show()

        def finalize_selection(event):
            if event.key != "enter":
                return
            plt.close()

        fig = plt.gcf()
        fig.canvas.mpl_connect('button_release_event', mouse_released)
        fig.canvas.mpl_connect('key_press_event', finalize_selection)

        plt.show()

    def select_coords(self):
        """
        GUI for manual selection of
        - Inner rectangle via its top left and bottom right points
        - vanishing point
        """
        self.select_rectangle()
        self.select_vanishing_point()

    def expand(self):
        """
        Given the user-specified points, this function expands the image to
        make each of the planar faces (ceiling, floor, left, right, back)
        are proper rectangles.
        :return (ceil_x, ceil_y, floor_x, floor_y, left_x, left_y, right_x, right_y, back_x, back_y)
        """
        y_max, x_max, channels = self.im.shape
        left_margin = -min(self.orx)
        right_margin = max(self.orx) - x_max
        top_margin = -min(self.ory)
        bottom_margin = max(self.ory) - y_max

        big_im_rgb = np.zeros((y_max + top_margin + bottom_margin, x_max + left_margin + right_margin, channels))
        big_im_rgb[top_margin: y_max + top_margin, left_margin: x_max + left_margin, :] = self.im
        big_im_alpha = np.expand_dims(np.zeros_like(big_im_rgb)[:, :, 0], 2)
        big_im_alpha[top_margin: y_max + top_margin, left_margin: x_max + left_margin] = 1
        big_im = np.append(big_im_rgb, big_im_alpha, axis=2)

        # update all variables
        vx = self.vx + left_margin
        vy = self.vy + top_margin
        irx = self.irx + left_margin
        iry = self.iry + top_margin
        orx = self.orx + left_margin
        ory = self.ory + top_margin

        ### define the 5 rectangles ###
        # ceiling
        ceil_x = [orx[0], orx[1], irx[1], irx[0]]
        ceil_y = [ory[0], ory[1], iry[1], iry[0]]
        if ceil_y[0] < ceil_y[1]:
            ceil_x[0] = find_line(vx, vy, ceil_x[0], ceil_y[0], ceil_y[1])[0]
            ceil_y[0] = ceil_y[1]
        else:
            ceil_x[1] = find_line(vx, vy, ceil_x[1], ceil_y[1], ceil_y[0])[0]
            ceil_y[1] = ceil_y[0]

        # floor
        floor_x = [irx[3], irx[2], orx[2], orx[3]]
        floor_y = [iry[3], iry[2], ory[2], ory[3]]
        if floor_y[2] > floor_y[3]:
            floor_x[2] = find_line(vx, vy, floor_x[2], floor_y[2], floor_y[3])[0]
            floor_y[2] = floor_y[3]
        else:
            floor_x[3] = find_line(vx, vy, floor_x[3], floor_y[3], floor_y[2])[0]
            floor_y[3] = floor_y[2]

        # left
        left_x = [orx[0], irx[0], irx[3], orx[3]]
        left_y = [ory[0], iry[0], iry[3], ory[3]]
        if left_x[0] < left_x[3]:
            left_y[0] = find_line(vx, vy, left_x[0], left_y[0], left_x[3])[1]
            left_x[0] = left_x[3]
        else:
            left_x[3] = find_line(vx, vy, left_x[3], left_y[3], left_x[0])[1]
            left_x[3] = left_x[0]

        # right
        right_x = [irx[1], orx[1], orx[2], irx[2]]
        right_y = [iry[1], ory[1], ory[2], iry[2]]
        if right_x[1] > right_x[2]:
            right_y[1] = find_line(vx, vy, right_x[1], right_y[1], right_x[2])[1]
            right_x[1] = right_x[2]
        else:
            right_y[2] = find_line(vx, vy, right_x[2], right_y[2], right_x[1])[1]
            right_x[2] = right_x[1]

        # back
        back_x = irx[:4]
        back_y = iry[:4]

        return big_im, ceil_x, ceil_y, floor_x, floor_y, left_x, left_y, right_x, right_y, back_x, back_y
    