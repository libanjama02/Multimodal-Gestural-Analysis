import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import time

while not os.path.exists("testforvisualization4.csv"):
    time.sleep(0.1)

import matplotlib
matplotlib.use('TkAgg')


class IncrementalDataReader:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, "r")
        self.columns = next(self.file).strip().split(",")

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.file)
        return dict(zip(self.columns, map(float, line.strip().split(","))))

    def close(self):
        self.file.close()


class Quaternion:
    """Quaternions for 3D rotations"""

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)

    @classmethod
    def from_v_theta(cls, v, theta):
        """
        Construct quaternion from unit vector v and rotation angle theta
        """
        theta = np.asarray(theta)
        v = np.asarray(v)

        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)
        vnrm = np.sqrt(np.sum(v * v))

        q = np.concatenate([[c], s * v / vnrm])
        return cls(q)

    @classmethod
    def from_q(cls, q):
        return cls(q)

    def __repr__(self):
        return "Quaternion:\n" + self.x.__repr__()

    def __mul__(self, other):
        # multiplication of two quaternions.
        prod = self.x[:, None] * other.x

        return self.__class__([(prod[0, 0] - prod[1, 1]
                                - prod[2, 2] - prod[3, 3]),
                               (prod[0, 1] + prod[1, 0]
                                + prod[2, 3] - prod[3, 2]),
                               (prod[0, 2] - prod[1, 3]
                                + prod[2, 0] + prod[3, 1]),
                               (prod[0, 3] + prod[1, 2]
                                - prod[2, 1] + prod[3, 0])])

    def as_v_theta(self):
        """Return the v, theta equivalent of the (normalized) quaternion"""
        # compute theta
        norm = np.sqrt((self.x ** 2).sum(0))
        if norm == 0.0:
            return np.asarray([0.0, 0.0, 0.0], dtype=float), 0.0
        theta = 2 * np.arccos(self.x[0] / norm)

        # compute the unit vector
        v = np.array(self.x[1:], order='F', copy=True)
        v /= np.sqrt(np.sum(v ** 2, 0))

        return v, theta

    def as_rotation_matrix(self):
        """Return the rotation matrix of the (normalized) quaternion"""
        v, theta = self.as_v_theta()
        if np.sqrt((v**2).sum(0)) == 0.0 and theta == 0.0:
            #print("v and theta are zero")
            return np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float)

        c = np.cos(theta)
        s = np.sin(theta)

        return np.array([[v[0] * v[0] * (1. - c) + c,
                          v[0] * v[1] * (1. - c) - v[2] * s,
                          v[0] * v[2] * (1. - c) + v[1] * s],
                         [v[1] * v[0] * (1. - c) + v[2] * s,
                          v[1] * v[1] * (1. - c) + c,
                          v[1] * v[2] * (1. - c) - v[0] * s],
                         [v[2] * v[0] * (1. - c) - v[1] * s,
                          v[2] * v[1] * (1. - c) + v[0] * s,
                          v[2] * v[2] * (1. - c) + c]])

class CubeAxes(plt.Axes):
    # colors of the faces
    colors = ['blue', 'green', 'white', 'yellow', 'orange', 'red']

    def __init__(self, fig, rect=[0, 0, 1, 1], *args, **kwargs):
        # We want to set a few of the arguments
        kwargs.update(dict(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), frameon=False,
                           xticks=[], yticks=[], aspect='equal'))
        super(CubeAxes, self).__init__(fig, rect, *args, **kwargs)

        # fiducial face is perpendicular to z at z=+1
        self.one_face = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, 1]])

        # construct six rotators for the face
        self.x, self.y, self.z = np.eye(3)
        self.rots = [Quaternion.from_v_theta(self.x, theta) for theta in (np.pi / 2, -np.pi / 2)]
        self.rots += [Quaternion.from_v_theta(self.y, theta) for theta in (np.pi / 2, -np.pi / 2)]
        self.rots += [Quaternion.from_v_theta(self.z, theta) for theta in (np.pi, 0)]

        self.xaxis.set_major_formatter(plt.NullFormatter())
        self.yaxis.set_major_formatter(plt.NullFormatter())

        # define the current rotation
        self.current_rot = Quaternion.from_v_theta((1, 1, 0), np.pi / 6)

    def draw_cube(self):
        """draw a cube rotated by theta around the given vector"""
        # rotate the six faces
        Rs = [(self.current_rot * rot).as_rotation_matrix() for rot in self.rots]
        faces = [np.dot(self.one_face, R.T) for R in Rs]

        # project the faces: we'll use the z coordinate
        # for the z-order
        faces_proj = [face[:, :2] for face in faces]
        zorder = [face[:4, 2].sum() for face in faces]

        # create the polygons if needed.
        # if they're already drawn, then update them
        if not hasattr(self, '_polys'):
            self._polys = [plt.Polygon(faces_proj[i], fc=self.colors[i],
                                       alpha=0.9, zorder=zorder[i])
                           for i in range(6)]
            for i in range(6):
                self.add_patch(self._polys[i])
        else:
            for i in range(6):
                self._polys[i].set_xy(faces_proj[i])
                self._polys[i].set_zorder(zorder[i])

        self.figure.canvas.draw()


class CubeAxesAuto(CubeAxes):
   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active = True 

    def start(self, event):
        self._active = not self._active
        while self._active:
            try:
                row = next(self.data_iter)
            except StopIteration:
                # No more data, wait for a moment and continue
                time.sleep(0.1)
                continue

            q = [row['w'], row['x'], row['y'], row['z']]
            if all(val == 0.0 for val in q):
                continue
            self.current_rot = Quaternion.from_q(q)
            self.draw_cube()

fig = plt.figure(figsize=(5, 5))
ax = CubeAxesAuto(fig, rect=[0.1, 0.2, 0.8, 0.7])
fig.add_axes(ax)
ax.draw_cube()

#reads data incrementally from the CSV file
data_iter = iter(IncrementalDataReader("testforvisualization4.csv"))
ax.data_iter = data_iter

# adds a pause button
pause_ax = fig.add_axes([0.4, 0.05, 0.2, 0.075])
pause_button = Button(pause_ax, 'Pause/Play')
pause_button.on_clicked(ax.start)

print("yes")