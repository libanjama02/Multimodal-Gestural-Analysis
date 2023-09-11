#import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider
import pandas as pd
from matplotlib.animation import FuncAnimation

# Read your dataframe
df_handpose = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\!twisthand_dataframe.csv")

# Defining the connections between landmarks (based on mediapipes framework )
connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5,9), (9, 10),
               (10, 11), (11, 12), (9,13), (13, 14), (14, 15), (15, 16), (13,17), (0, 17), (17, 18),
               (18, 19), (19, 20)]

# initializing the 3D scatter plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], c='r', marker='o')

# drawing the connections
lines = [ax.plot([], [], [], 'b')[0] for _ in connections]

def update(frame):
    x_coords = df_handpose.loc[frame, ['x_' + str(i) for i in range(21)]].values
    y_coords = df_handpose.loc[frame, ['y_' + str(i) for i in range(21)]].values
    z_coords = df_handpose.loc[frame, ['z_' + str(i) for i in range(21)]].values
    
    # iteratively updating the scatter plot data
    sc._offsets3d = (x_coords, y_coords, z_coords)
    
    # same for line connections
    for i, connection in enumerate(connections):
        lines[i].set_data([x_coords[connection[0]], x_coords[connection[1]]],
                          [y_coords[connection[0]], y_coords[connection[1]]])
        lines[i].set_3d_properties([z_coords[connection[0]], z_coords[connection[1]]])
    
    # Setting axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    progress_bar.set_val(frame)

# Attempt at play/pause (matplotlib seems to struggle to start the animation paused no matter what i've tried)
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

# play/pause and slider code 
play_pause_ax = fig.add_axes([0.35, 0.02, 0.3, 0.04])
play_pause_button = Button(play_pause_ax, 'Play', color='lightgray')
play_pause_button.on_clicked(on_play_pause)

progress_ax = fig.add_axes([0.1, 0.06, 0.8, 0.02], facecolor='lightgray')
progress_bar = Slider(progress_ax, 'Progress', 0, len(df_handpose)-1, valinit=0, valstep=1, valfmt='%i')
progress_bar.slidermin = 0
progress_bar.slidermax = len(df_handpose)-1

#initializing the animation
anim = FuncAnimation(fig, update, frames=len(df_handpose), repeat=False, interval=100)
anim.event_source.stop()

plt.show()
