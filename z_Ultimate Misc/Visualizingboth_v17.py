import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
# import matplotlib.animation as animation
# import mpl_toolkits.mplot3d.art3d as art3d
from scipy.spatial.transform import Rotation as R
from itertools import combinations, product 
# import matplotlib.patches as patches
import ast
from matplotlib.widgets import Button, Slider

# Function to convert strings to lists, interpreting 'nan' strings as actual NaN values
def string_to_list(s):
    try:
        return ast.literal_eval(s) #converts to list
    except ValueError:  # 'nan'
        return None

# Cube Class
class Cube:
    def __init__(self, edge_length=1, position=(0,0,0), rotation_matrix=None):
        self.edge_length = edge_length
        self.position = position
        self.rotation_matrix = rotation_matrix

    def draw(self, ax):
        r = [-self.edge_length / 2, self.edge_length / 2]
        for s, e in combinations(np.array(list(product(r, r, r))), 2):
            s, e = np.array(s), np.array(e)
            if np.sum(s != e) == 1:
                if self.rotation_matrix is not None:
                    s = np.dot(self.rotation_matrix, s)
                    e = np.dot(self.rotation_matrix, e)
                ax.plot3D(*zip(s+self.position, e+self.position), color="b")

# Converts 4d quaternion to 3d rotation matrix within the update function
def quaternion_to_rotation(quaternion):
    r = R.from_quat(quaternion)
    return r.as_matrix()

# Load the dataframe
df = pd.read_pickle(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\preprocessed_dataframe_filled_v4.pkl")
# Applies lambda function. x is the individaul element within a column. If nan, we convert x to 0. Otherwise, calls string to list funct
df['Parsed Hand Pose Data'] = df['Parsed Hand Pose Data'].apply(lambda x: string_to_list(x) if x != 'nan' else [0.0]*63)
df['Wrist'] = df['Wrist'].apply(lambda x: string_to_list(x) if x != 'nan' else [0.0]*3) 

hand_pose = df['Parsed Hand Pose Data'].tolist()
quaternion = df['Wrist'].tolist()

# Debug:  Checks if a list is empty
def is_empty(lst):
    print(f"Checking if {lst} is empty...")  # Adds this line to terminal
    return lst is None or len(lst) == 0


# Debug: Check for empty lists in 'Parsed Hand Pose Data'
is_empty_hand_pose_data = df['Parsed Hand Pose Data'].apply(is_empty)
empty_rows_hand_pose_data = df[is_empty_hand_pose_data]
print(f"Number of rows with empty lists in 'Parsed Hand Pose Data': {len(empty_rows_hand_pose_data)}")

# Debug: Check for empty lists in 'Wrist' 
is_empty_wrist = df['Wrist'].apply(is_empty)
empty_rows_wrist = df[is_empty_wrist]
print(f"Number of rows with empty lists in 'Wrist': {len(empty_rows_wrist)}")

# Debug: Prints index and step count.
print(df.index)

# Making an instance of the cube class (edit bracket inside to change size)
cube = Cube(0.5)

# skeletal connections between landmarks 
connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
               (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
               (18, 19), (19, 20)]

#Sets figure size in inches
fig = plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=0.2) 
ax = fig.add_subplot(111, projection='3d')

# Function that is called for each frame of animation
def update(num):
    print(f"Processing frame {num}...")  # For debugging
    ax.cla() #clears axis 
    ax.set_xlim3d(-1, 1) 
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    # Adjusts the frame ID to start from 0 because current first frame is 3 in dataframe
    adjusted_num = num + 3

    # Retrieve hand pose and quaternion data for this frame
    hand_pose = df.iloc[num]['Parsed Hand Pose Data']
    quaternion = df.iloc[num][['w', 'x', 'y', 'z']].tolist() 

    # If hand pose data or quaternion data is None or an empty list, skip this frame
    if (hand_pose is None or len(hand_pose) == 0) or (quaternion is None or len(quaternion) == 0):
        print(f"Skipping frame {adjusted_num} due to no data. Hand pose: {hand_pose}, Quaternion: {quaternion}")
        return

    # If hand pose data or quaternion data is NaN, skip this frame
    if any(pd.isna(val) for val in hand_pose) or any(pd.isna(val) for val in quaternion):
        print(f"Skipping frame {adjusted_num} due to NaN values.")
        return

    # Plots the wrist landmark (landmark 0).
    ax.scatter(quaternion[0], quaternion[1], quaternion[2], color='red', s=40) #s = size of dot

    # Plots the hand landmarks. ::3 = starts at 0th element, goes to end, selects every third element = x
    #1::3 starts at first element, goes to end, selects every three elements = y, and so on
    ax.scatter(hand_pose[::3], hand_pose[1::3], hand_pose[2::3])
    for connection in connections:   
        start = connection[0] * 3
        end = connection[1] * 3
        ax.plot(hand_pose[start:end:3],
                hand_pose[start + 1:end + 1:3],
                hand_pose[start + 2:end + 2:3], 'gray')

    # Rotate cube
    cube.rotation_matrix = quaternion_to_rotation(quaternion)
    cube.position = quaternion[0:3]  # This line is added to set the cube's position to landmark 0's position (changed hand_pose to quaternion here which seems to fix it)
    

    print(f'Cube position for frame {adjusted_num}:', cube.position)
    print(f'Hand landmarks for frame {adjusted_num}:', hand_pose)
    print(f'Quaternion for frame {adjusted_num}:', quaternion)

    cube.draw(ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=-90, elev=-90) #azimuth angle and elevation in z direction
    
#calls update func, animates to the len of hand pose which is 158, each frame displayed for 100 milliseconds
ani = FuncAnimation(fig, update, frames=len(hand_pose), interval=100)

# Used to control the state of the animation WIP
ani.running = True

# Function to start/stop the animation when the button is clicked
def toggle_animation(event):
    if ani.running:
        ani.event_source.stop()
    else:
        ani.event_source.start()
    ani.running = not ani.running

# Function to update the frame number when the slider is moved WIP
def update_frame_number(val):
    ani.frame_seq = ani.new_frame_seq()
    ani.event_source.interval = val

# Create the play/pause button WIP
pause_ax = fig.add_axes([0.4, 0.05, 0.2, 0.075])
pause_button = Button(pause_ax, 'Pause/Play')
pause_button.on_clicked(toggle_animation)

# Create the progress slider WIP
progress_ax = fig.add_axes([0.1, 0.15, 0.8, 0.03])
progress_slider = Slider(progress_ax, 'Progress', 0, len(hand_pose), valinit=0, valstep=1)
progress_slider.on_changed(update_frame_number)

plt.show()



