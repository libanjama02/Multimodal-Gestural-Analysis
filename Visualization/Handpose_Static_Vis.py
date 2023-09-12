#debugging script
import matplotlib.pyplot as plt
import pandas as pd

# reading the dataframe
df_handpose = pd.read_csv(r"C:\Users\liban\OneDrive - King's College London\!library_placementnotes\Project\Experiments\twisthand\!twisthand_dataframe.csv")

# Extracting the x, y, z coordinates for the 21 landmarks for Frame 0. Feel free to change
x_coords = df_handpose.loc[0, ['x_' + str(i) for i in range(21)]].values
y_coords = df_handpose.loc[0, ['y_' + str(i) for i in range(21)]].values
z_coords = df_handpose.loc[0, ['z_' + str(i) for i in range(21)]].values

# Connections based on mediapipes method

connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5,9), (9, 10),
               (10, 11), (11, 12), (9,13), (13, 14), (14, 15), (15, 16), (13,17), (0, 17), (17, 18),
               (18, 19), (19, 20)]

# plotting the landmarks with connections
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')

for connection in connections:
    ax.plot([x_coords[connection[0]], x_coords[connection[1]]],
            [y_coords[connection[0]], y_coords[connection[1]]],
            [z_coords[connection[0]], z_coords[connection[1]]], 'b')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

plt.show()
