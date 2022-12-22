from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pickle
from image_tools import *
import vpython as vp
from vpython import canvas

vp.scene.width = 600
vp.scene.height = 600
vp.scene.camera.pos = vp.vec(0, 1, 0)
vp.scene.camera.axis = vp.vec(0, 0, 1)


def find_line(x1, y1, x2, y2, c):
    # Given two points (x1, y1) and (x2, y2), find a fitting line L
    # Then, return x such that L(x) = c  and y such that L(c) = y
    m = (y1 - y2) / (x1 - x2)
    b = y1 - m * x1
    return (c - b) / m, m * c + b


def render_plane(vertices, texture):
    # vertices will be a list of 3-tuples (or a 3d array works)
    # texture is some img array
    fname = f"lib/textures/texture{hash(str(vertices))}.jpg"
    plt.imsave(fname, texture)  # might need a better name idk
    vs = []
    texposes = [vp.vec(0, 0, 0), vp.vec(1, 0, 0), vp.vec(1, 1, 0), vp.vec(0, 1, 0)]
    for i in range(4):
        v = vp.vec(*vertices[i])
        vs.append(vp.vertex(pos=v, texpos=texposes[i]))

    vp.quad(vs=vs, texture=fname, shininess=0)


def find_corner(vx, vy, rx, ry, max_x, max_y):
    # given a line thru (vx, vy) and (rx, ry), find
    # intersection with image (x,y)
    y1 = max_y
    x1 = find_line(vx, vy, rx, ry, max_y)[0]
    x2 = max_x
    y2 = find_line(vx, vy, rx, ry, max_x)[1]
    if np.linalg.norm([vx - x1, vy - y1]) > np.linalg.norm([vx - x2, vy - y2]):
        return x1, y1
    else:
        return x2, y2


def calculate_op(im, vx, vy, irx, iry):
    # Find where the line from VP through inner rectangle hits the edge of image
    y_max, x_max, _ = im.shape

    opx, opy = [], []
    x_bounds = [0, x_max, x_max, 0]
    y_bounds = [0, 0, y_max, y_max]
    for i in range(4):
        x, y = find_corner(vx, vy, irx[i], iry[i], x_bounds[i], y_bounds[i])
        opx.append(int(x))
        opy.append(int(y))
    opx = np.array(opx)
    opy = np.array(opy)

    return opx, opy


def plot_planar_lines(im, irx, iry, vx, vy):
    opx, opy = calculate_op(im, vx, vy, irx, iry)
    plt.plot(irx, iry, 'b')  # rectangle
    plt.plot(vx, vy, 'b')  # vanishing point
    for i in range(4):  # planar lines
        plt.plot([vx, irx[i]], [vy, iry[i]], 'r-.')
        plt.plot([opx[i], irx[i]], [opy[i], iry[i]], 'r')
        plt.draw()
    plt.imshow(im)
    plt.figtext(0.5, 0.05, "Select vanishing point. Finalize with \n ENTER. Fullscreen recommended.", wrap=True,
                horizontalalignment='center', fontsize=10, c='r', backgroundcolor='k')
    return opx, opy


def plot_rectangle(rx, ry, c):
    plt.plot(np.append(rx, rx[0]), np.append(ry, ry[0]), c)


@dataclass
class Model3d:
    image: np.ndarray
    bounds: np.ndarray
    ceil_x: np.ndarray
    ceil_y: np.ndarray
    floor_x: np.ndarray
    floor_y: np.ndarray
    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray
    back_x: np.ndarray
    back_y: np.ndarray
    vertices: np.ndarray = np.zeros((3, 12))  # 12 vertices in 3D
    depth: float = 1000  # depth in the z direction scale factor
    f: float = 1  # focal length
    scale: float = 1000  # pixel density / image quality
    o_x: float = None
    o_y: float = None

    def __post_init__(self):
        # here is an ascii diagram
        """
            11
              xx
              xxx
              x  xx
              x   xxx 8                 9
              x     xxxxxxxxxxxxxxxxxxxx
              x       xx              xx
              x        xx ceil     xxxx 10
              x         xx      xxx   x
              x        6 xxxxxxxx 7   x
              x          x back x     x
              x          x      x right
              x left   0 xxxxxxxx 1   x
              x          x     xx     x
              x         xx floor x    x
              x       xxx        xx   x
              x      xxxxxxxxxxxxxxxxxx
              x     xx 3          2 xxx
              x   xx                 xx
              x  xx                   5
              xxxx
              xx
             4
        """
        ### determine o_x, o_y ###
        self.determine()

        ### place vertices corresponding at y = 0 (0,1,2,3,4,5) ###
        screen_x = self.pixel2imgscreen_x(np.append(self.floor_x, [self.left_x[3], self.right_x[2]]))
        screen_y = self.pixel2imgscreen_y(np.append(self.floor_y, [self.left_y[3], self.right_y[2]]))
        x_floor = screen_x / (1 - screen_y)
        y_floor = [0] * 6
        z_floor = self.f / (screen_y - 1)

        for i in range(6):
            self.vertices[:, i] = x_floor[i], y_floor[i], z_floor[i]

        ### place vertices corresponding to back wall (6, 7) ###
        screen_x = self.pixel2imgscreen_x(self.back_x[:2])
        screen_y = self.pixel2imgscreen_y(self.back_y[:2])
        d = self.vertices[2, 0]  # z coordinate of first vertex
        x_floor = - screen_x * d / self.f
        y_floor = 1 + (d / self.f) * (1 - screen_y)
        z_floor = [d, d]

        for i in range(2):
            self.vertices[:, i + 6] = x_floor[i], y_floor[i], z_floor[i]

        ### place vertices on roof (8, 9, 10, 11) ###
        # remark; we do by a projection onto y = H
        # this may cause misalignment with the x coordinate of back wall
        # this issue is to be resolved later
        screen_x = self.pixel2imgscreen_x(np.append(self.ceil_x[:2], [self.right_x[1], self.left_x[0]]))
        screen_y = self.pixel2imgscreen_y(np.append(self.ceil_y[:2], [self.right_y[1], self.left_y[0]]))
        H = self.vertices[1, 6]  # y coordinate of 6th vertex
        frame = (H - screen_y) / (1 - screen_y)
        x_roof = screen_x * (1 - frame)
        y_roof = [H] * 4
        z_roof = self.f * (frame - 1)

        for i in range(4):
            self.vertices[:, i + 8] = x_roof[i], y_roof[i], z_roof[i]

        ### set 0, 3, 4, 6, 8, 11 to have same x value ###
        self.vertices[0, [0, 3, 4, 6, 8, 11]] = np.mean(self.vertices[0, [0, 3, 4, 6, 8, 11]])

        ### set 1, 2, 5, 7, 9, 10 to have same x value ###
        self.vertices[0, [1, 2, 5, 7, 9, 10]] = np.mean(self.vertices[0, [1, 2, 5, 7, 9, 10]])

        ## set 2, 3 & 4, 11 & 5, 10 & 8, 9 to have same z value ###
        # self.vertices[2, [2, 3]] = np.mean(self.vertices[2, [2, 3]])
        # self.vertices[2, [4, 11]] = np.mean(self.vertices[2, [4, 11]])
        # self.vertices[2, [5, 10]] = np.mean(self.vertices[2, [5, 10]])
        # self.vertices[2, [8, 9]] = np.mean(self.vertices[2, [8, 9]])


    def plot_rectangles(self):
        plot_rectangle(self.back_x, self.back_y, 'w*')
        plot_rectangle(self.ceil_x, self.ceil_y, 'r-')
        plot_rectangle(self.right_x, self.right_y, 'r-')
        plot_rectangle(self.left_x, self.left_y, 'r-')
        plot_rectangle(self.floor_x, self.floor_y, 'r-')

    def determine(self):
        u1, u2, u3, u4 = self.floor_x
        v1, _, v2, _ = self.floor_y
        p = (u4 * v1 - u1 * v2 + u2 * v2 - u3 * v1) / (u1 - u4 - u2 + u3)
        self.o_y = -p - (self.depth / self.f)
        self.o_x = -(u4 * v1 - u1 * v2 - p * (u1 - u4)) / (v2 - v1)

    def pixel2imgscreen_x(self, x_coords):
        return self.f * (x_coords - self.o_x) / self.depth

    def pixel2imgscreen_y(self, y_coords):
        return self.f * (y_coords - self.o_y) / self.depth

    def set_vertex(self, vertex_idx, x_list, y_list, z_list, list_idx):
        self.vertices[vertex_idx] = x_list[list_idx], y_list[list_idx], z_list[list_idx]

    def ceil(self):
        ceil_vertices = self.vertices[:, [6, 7, 9, 8]]
        ceil_correspondences = np.array([
            self.ceil_y,
            self.ceil_x,
            [1, 1, 1, 1],
            ceil_vertices[2],
            ceil_vertices[0],
            [1, 1, 1, 1]
        ])
        flipped_vertices = ceil_vertices[:, [2, 3, 0,1]]
        # np.array([ceil_vertices[:, 2], ceil_vertices[3], ceil_vertices[0], ceil_vertices[1]])
        return flipped_vertices, ceil_correspondences

    def floor(self):
        floor_vertices = self.vertices[:, [0, 1, 2, 3]]
        floor_correspondences = np.array([
            self.floor_y,
            self.floor_x,
            [1, 1, 1, 1],
            floor_vertices[2],
            floor_vertices[0],
            [1, 1, 1, 1]
        ])
        # floor needs flip orientation
        flipped_vertices = self.vertices[:, [1, 0, 3, 2]]
        return flipped_vertices, floor_correspondences

    def left(self):
        left_vertices = self.vertices[:, [4, 0, 6, 11]]
        print(left_vertices)
        left_correspondences = np.array([
            self.left_y,
            self.left_x,
            [1, 1, 1, 1],
            left_vertices[1],
            left_vertices[2],
            [1, 1, 1, 1]
        ])
        return left_vertices, left_correspondences

    def right(self):
        right_vertices = self.vertices[:, [1, 5, 10, 7]]
        right_correspondences = np.array([
            self.right_y,
            self.right_x,
            [1, 1, 1, 1],
            right_vertices[1],
            right_vertices[2],
            [1, 1, 1, 1]
        ])
        # leave this as is, I think
        flipped_vertices = self.vertices[:, [5, 1, 7, 10]]
        return flipped_vertices, right_correspondences

    def back(self):

        back_vertices = self.vertices[:, [6, 7, 1, 0]]
        back_correspondences = np.array([
            self.back_y,
            self.back_x,
            [1, 1, 1, 1],
            back_vertices[1],
            back_vertices[0],
            [1, 1, 1, 1]
        ])
        flipped_vertices = self.vertices[:, [7, 6, 0, 1]]
        return flipped_vertices, back_correspondences

    def render(self):
        for vs, cs in [self.ceil(), self.floor(), self.left(), self.right(), self.back()]:
            cs[[3, 4], :] = cs[[3, 4], :] * self.scale
            vs_tuples = list(map(tuple, vs.T.tolist()))

            texture = warp2(self.image, cs)

            render_plane(vs_tuples, texture)


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

    def show(self, save=False):
        plt.imshow(self.im)
        if save:
            plt.savefig(save)
        plt.show()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

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
            x, y = self.find_corner(i)
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

    def get_model(self, depth=1000, scale=1000):
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
        bounds = np.array([left_margin, top_margin, x_max + left_margin, y_max + top_margin])

        big_im = np.zeros((y_max + top_margin + bottom_margin, x_max + left_margin + right_margin, channels))
        big_im[top_margin: y_max + top_margin, left_margin: x_max + left_margin, :] = self.im
        big_im_alpha = np.expand_dims(np.zeros_like(big_im)[:, :, 0], 2)
        big_im_alpha[top_margin: y_max + top_margin, left_margin: x_max + left_margin] = 1
        im = np.append(big_im, big_im_alpha, axis=2)

        # update all variables
        vx = self.vx + left_margin
        vy = self.vy + top_margin
        irx = self.irx + left_margin
        iry = self.iry + top_margin
        orx = self.orx + left_margin
        ory = self.ory + top_margin

        ### define the 5 rectangles ###
        # ceiling
        ceil_x = np.array([orx[0], orx[1], irx[1], irx[0]])
        ceil_y = np.array([ory[0], ory[1], iry[1], iry[0]])
        if ceil_y[0] < ceil_y[1]:
            ceil_x[0] = find_line(vx, vy, ceil_x[0], ceil_y[0], ceil_y[1])[0]
            ceil_y[0] = ceil_y[1]
        else:
            ceil_x[1] = find_line(vx, vy, ceil_x[1], ceil_y[1], ceil_y[0])[0]
            ceil_y[1] = ceil_y[0]

        # floor
        floor_x = np.array([irx[3], irx[2], orx[2], orx[3]])
        floor_y = np.array([iry[3], iry[2], ory[2], ory[3]])
        if floor_y[2] > floor_y[3]:
            floor_x[2] = find_line(vx, vy, floor_x[2], floor_y[2], floor_y[3])[0]
            floor_y[2] = floor_y[3]
        else:
            floor_x[3] = find_line(vx, vy, floor_x[3], floor_y[3], floor_y[2])[0]
            floor_y[3] = floor_y[2]

        # left
        left_x = np.array([orx[0], irx[0], irx[3], orx[3]])
        left_y = np.array([ory[0], iry[0], iry[3], ory[3]])
        if left_x[0] < left_x[3]:
            left_y[0] = find_line(vx, vy, left_x[0], left_y[0], left_x[3])[1]
            left_x[0] = left_x[3]
        else:
            left_x[3] = find_line(vx, vy, left_x[3], left_y[3], left_x[0])[1]
            left_x[3] = left_x[0]

        # right
        right_x = np.array([irx[1], orx[1], orx[2], irx[2]])
        right_y = np.array([iry[1], ory[1], ory[2], iry[2]])
        if right_x[1] > right_x[2]:
            right_y[1] = find_line(vx, vy, right_x[1], right_y[1], right_x[2])[1]
            right_x[1] = right_x[2]
        else:
            right_y[2] = find_line(vx, vy, right_x[2], right_y[2], right_x[1])[1]
            right_x[2] = right_x[1]

        # back
        back_x = irx[:4]
        back_y = iry[:4]

        return Model3d(im, bounds, ceil_x, ceil_y, floor_x, floor_y, left_x, left_y, right_x, right_y, back_x, back_y, depth=depth, scale=scale)
