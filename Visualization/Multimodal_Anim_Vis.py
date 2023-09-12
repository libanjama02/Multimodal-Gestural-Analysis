import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button, Slider
from matplotlib.gridspec import GridSpec
import pandas as pd
from matplotlib.animation import FuncAnimation

df_handpose = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\!twisthand_dataframe.csv")

# using mediapipe's framework to define the connections
connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5,9), (9, 10),
               (10, 11), (11, 12), (9,13), (13, 14), (14, 15), (15, 16), (13,17), (0, 17), (17, 18),
               (18, 19), (19, 20)]

#Setting up the figure and the subplots using GridSpec
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(1, 2, width_ratios=[1, 1])

ax_hand = fig.add_subplot(gs[0], projection='3d')  # For hand pose
ax_quat = fig.add_subplot(gs[1], projection='3d')  # For quaternion cube

sc = ax_hand.scatter([], [], [], c='r', marker='o')

# drawing the connections for hand pose
lines = [ax_hand.plot([], [], [], 'b')[0] for _ in connections]

# same quat class as before
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


# Defining a function to draw the cube with colored faces
def draw_cube(ax, rotation_matrix, colors=['blue', 'green', 'white', 'yellow', 'orange', 'red']):
    # Defining the cube vertices
    points = np.array([[-0.15, -0.15, -0.15],
                       [0.15, -0.15, -0.15],
                       [0.15, 0.15, -0.15],
                       [-0.15, 0.15, -0.15],
                       [-0.15, -0.15, 0.15],
                       [0.15, -0.15, 0.15],
                       [0.15, 0.15, 0.15],
                       [-0.15, 0.15, 0.15]])
    
    #applying a transformation matrix to fix the orientation to be accurate to Mbientlab IMU orientation (-1 in Z axis)

    M = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, -1]])

    accurate_rotation_matrix = np.dot(M, rotation_matrix)

    # Apply the rotation
    points = np.dot(points, accurate_rotation_matrix.T)

    # Defining the vertices that compose each face of the cube
    Z = [[points[j] for j in [0, 1, 5, 4]],
         [points[j] for j in [7, 6, 2, 3]],
         [points[j] for j in [0, 3, 2, 1]],
         [points[j] for j in [7, 4, 5, 6]],
         [points[j] for j in [7, 3, 0, 4]],
         [points[j] for j in [2, 6, 5, 1]]]
    
    # Drawing the cube faces
    ax.add_collection3d(Poly3DCollection(Z, facecolors=colors, linewidths=1, edgecolor='k'))

#This function updates each frame of the animation from sequential data in the dataframe
def update(frame):
    x_coords = df_handpose.loc[frame, ['x_' + str(i) for i in range(21)]].values
    y_coords = df_handpose.loc[frame, ['y_' + str(i) for i in range(21)]].values
    z_coords = df_handpose.loc[frame, ['z_' + str(i) for i in range(21)]].values
    
    sc._offsets3d = (x_coords, y_coords, z_coords)
    
    for i, connection in enumerate(connections):
        lines[i].set_data([x_coords[connection[0]], x_coords[connection[1]]],
                          [y_coords[connection[0]], y_coords[connection[1]]])
        lines[i].set_3d_properties([z_coords[connection[0]], z_coords[connection[1]]])

    #Update the quaternion cube visualization
    ax_quat.cla()
    #Extracting the quaternion data from the dataframe
    q = df_handpose.loc[frame, ['w', 'x', 'y', 'z']].values
    current_rot = Quaternion.from_q(q)
    rotation_matrix = current_rot.as_rotation_matrix()
    draw_cube(ax_quat, rotation_matrix)

    # Setting axis limits for better viewing
    ax_quat.set_xlim([-0.5, 0.5])
    ax_quat.set_ylim([-0.5, 0.5])
    ax_quat.set_zlim([-0.5, 0.5])

    # defining the labels
    ax_hand.set_xlabel('X Axis')
    ax_hand.set_ylabel('Y Axis')
    ax_hand.set_zlabel('Z Axis')
    ax_quat.set_xlabel('X Axis')
    ax_quat.set_ylabel('Y Axis')
    ax_quat.set_zlabel('Z Axis')

    progress_bar.set_val(frame)

# Same play/pause button as before
playing = False
def on_play_pause(event):
    global playing
    if playing:
        anim.event_source.stop()
        play_pause_button.label.set_text('Play')
        playing = False
    else:
        anim.event_source.start()
        play_pause_button.label.set_text('Pause')
        playing = True

# code for slider and pause button, same issues as before
play_pause_ax = fig.add_axes([0.35, 0.02, 0.3, 0.04])
play_pause_button = Button(play_pause_ax, 'Play', color='lightgray')
play_pause_button.on_clicked(on_play_pause)

progress_ax = fig.add_axes([0.1, 0.06, 0.8, 0.02], facecolor='lightgray')
progress_bar = Slider(progress_ax, 'Progress', 0, len(df_handpose)-1, valinit=0, valstep=1, valfmt='%i')

# intended to initialize at a paused state. doesn't really work. 
anim = FuncAnimation(fig, update, frames=len(df_handpose), repeat=False, interval=100)
anim.event_source.stop()

plt.show()
