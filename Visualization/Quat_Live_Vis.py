from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from collections import deque

#structure to hold the latest quaternion value
quaternion_buffer = deque(maxlen=1)

class State:
    def __init__(self, device):
        self.device = device
        self.samples = 0
        self.callback = FnVoid_VoidP_DataP(self.data_handler)

    def data_handler(self, ctx, data):
        quat_str = "QUAT: %s -> %s" % (self.device.address, parse_value(data))
        print(quat_str)  
        self.samples += 1

        #Parsing the quaternion values and storing them in the buffer
        quat_values = [float(val.split(":")[1].strip()) for val in quat_str.split("{")[1].split("}")[0].split(",")]
        quaternion_buffer.append(quat_values)

#hardcoded MAC address 
mac_address = "D3:99:34:4D:01:CC"
device = MetaWear(mac_address)
device.connect()
print("Connected to " + device.address)

state = State(device)

print("Configuring device")
libmetawear.mbl_mw_settings_set_connection_parameters(state.device.board, 7.5, 7.5, 0, 6000)
sleep(1.5)

libmetawear.mbl_mw_sensor_fusion_set_mode(state.device.board, SensorFusionMode.NDOF)
libmetawear.mbl_mw_sensor_fusion_set_acc_range(state.device.board, SensorFusionAccRange._8G)
libmetawear.mbl_mw_sensor_fusion_set_gyro_range(state.device.board, SensorFusionGyroRange._2000DPS)
libmetawear.mbl_mw_sensor_fusion_write_config(state.device.board)

signal = libmetawear.mbl_mw_sensor_fusion_get_data_signal(state.device.board, SensorFusionData.QUATERNION)
libmetawear.mbl_mw_datasignal_subscribe(signal, None, state.callback)

class Quaternion:
    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)

    @classmethod
    def from_v_theta(cls, v, theta):
        theta = np.asarray(theta)
        v = np.asarray(v)
        s = np.sin(0.5 * theta)
        c = np.cos(0.5 * theta)
        vnrm = np.sqrt(np.sum(v * v))
        q = np.concatenate([[c], s * v / vnrm])
        return cls(q)

    def as_v_theta(self):
        """Return the v, theta equivalent of the (normalized) quaternion"""
        x = self.x / np.sqrt(np.sum(self.x * self.x))
        theta = 2 * np.arccos(x[0])
        v = x[1:] / np.sqrt(np.sum(x[1:] ** 2))
        return v, theta

    def as_rotation_matrix(self):
        """Return the rotation matrix of the (normalized) quaternion"""
        v, theta = self.as_v_theta()
        v /= np.sqrt(np.sum(v ** 2))
        c = np.cos(theta); s = np.sin(theta)
        return np.array([[v[0] * v[0] * (1 - c) + c,
                         v[0] * v[1] * (1 - c) - v[2] * s,
                         v[0] * v[2] * (1 - c) + v[1] * s],
                        [v[0] * v[1] * (1 - c) + v[2] * s,
                         v[1] * v[1] * (1 - c) + c,
                         v[1] * v[2] * (1 - c) - v[0] * s],
                        [v[0] * v[2] * (1 - c) - v[1] * s,
                         v[1] * v[2] * (1 - c) + v[0] * s,
                         v[2] * v[2] * (1 - c) + c]])

#Matplotlib setup for 3D visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def draw_cube(R):
    vertices = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]).T
    vertices_transformed = R @ vertices
    
    #Plot the transformed vertices to create the cube's edges
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    line_colors = ['red', 'red', 'red', 'red', 'purple', 'purple', 'purple', 'purple', 'green', 'black', 'orange', 'green']
    for line, color in zip(lines, line_colors):
        ax.plot3D(vertices_transformed[0, line], vertices_transformed[1, line], vertices_transformed[2, line], color)
       
    #Plot the vertices as blue dots(might change colour)
    ax.plot3D(vertices_transformed[0, :], vertices_transformed[1, :], vertices_transformed[2, :], 'bo')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def update_cube():
    if quaternion_buffer:
        quat_values = quaternion_buffer[-1]
        R = Quaternion(quat_values).as_rotation_matrix()
        ax.clear()
        draw_cube(R)
        plt.draw()


#Initiating the animation loop to update the cube's orientation
plt.ion()

#Starting the IMU data reading
libmetawear.mbl_mw_sensor_fusion_enable_data(state.device.board, SensorFusionData.QUATERNION)
libmetawear.mbl_mw_sensor_fusion_start(state.device.board)

#Main loop to keep the script running and update the visualization
try:
    while True:
        update_cube() # Update the cube's orientation based on the latest quaternion value
        plt.pause(0.005) # Pause to allow time for visualization updates
        sleep(0.01)  # Allow time for data reading and visualization updates
except KeyboardInterrupt:
    print("Stopping the script...")

#Clean up and disconnect the device
libmetawear.mbl_mw_sensor_fusion_stop(state.device.board)
libmetawear.mbl_mw_datasignal_unsubscribe(signal)
libmetawear.mbl_mw_debug_disconnect(state.device.board)
print("Disconnected from " + device.address)
