# WIP. To be used for live quaternion visualization.

from __future__ import print_function
from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
from time import sleep
from threading import Event
from optparse import OptionParser
import platform
import sys
import os
import time
from datetime import datetime 

class State:
    def __init__(self, device="", filename=""):
        self.device = device
        self.filename = filename
        self.samples = 0
        self.callback_all = FnVoid_VoidP_DataP(self.data_handler_all)
        self.data_quat = None
        self.data_acc = None
        # Initialize quaternion and acceleration values (but why, this isn't in  the original imu script?)
        self.qw = self.qx = self.qy = self.qz = None
        self.ax = self.ay = self.az = None
        with open(self.filename, "w") as file_stream:
            file_stream.write("Time,Quaternions, Acceleration\n")
        self.now = time.time()
        
    def data_handler_all(self, ctx, data):
        values = parse_value(data)
        try:
            values.w
            self.data_quat = values
            self.qw, self.qx, self.qy, self.qz = values.w, values.x, values.y, values.z
        except AttributeError:
            self.data_acc = values
            self.ax, self.ay, self.az = values.x, values.y, values.z
        time_again = time.time()
        elapsed_ms = round((time_again-self.now)*1000)
        self.samples+= 1

        # Write data to file only if both quaternion and acceleration data have been received
        if self.data_quat is not None and self.data_acc is not None:
            with open(self.filename, "a") as file_stream:
                file_stream.write(f"{datetime.utcnow().isoformat()},{self.qw},{self.qx},{self.qy},{self.qz},{self.ax},{self.ay},{self.az}\n")


# Hard-coded MAC address and filename for simplicity
mac_address = "D3:99:34:4D:01:CC"
filename = "testforvisualization3.csv"

#device instead of dl it seems.
device = MetaWear(mac_address)
device.connect()
state = State(device=device, filename=filename)

libmetawear.mbl_mw_settings_set_connection_parameters(state.device.board, 7.5, 7.5, 0, 6000)
sleep(1.5)

# ============= SETUP ============= 
libmetawear.mbl_mw_sensor_fusion_set_mode(state.device.board, SensorFusionMode.NDOF)
libmetawear.mbl_mw_sensor_fusion_set_acc_range(state.device.board, SensorFusionAccRange._8G)
libmetawear.mbl_mw_sensor_fusion_set_gyro_range(state.device.board, SensorFusionGyroRange._2000DPS)
libmetawear.mbl_mw_sensor_fusion_write_config(state.device.board)
libmetawear.mbl_mw_acc_set_odr(state.device.board, 100.0)
libmetawear.mbl_mw_acc_set_range(state.device.board, 16.0)
libmetawear.mbl_mw_acc_write_acceleration_config(state.device.board)

# get quat signal and subscribe
signal_lq = libmetawear.mbl_mw_sensor_fusion_get_data_signal(state.device.board, SensorFusionData.QUATERNION)
libmetawear.mbl_mw_datasignal_subscribe(signal_lq, None, state.callback_all)

# Get accelerometer data
signal_la = libmetawear.mbl_mw_acc_get_acceleration_data_signal(state.device.board)
libmetawear.mbl_mw_datasignal_subscribe(signal_la, None, state.callback_all)

# ============= START =============
libmetawear.mbl_mw_sensor_fusion_enable_data(state.device.board, SensorFusionData.QUATERNION)
libmetawear.mbl_mw_sensor_fusion_start(state.device.board)
libmetawear.mbl_mw_acc_enable_acceleration_sampling(state.device.board)
libmetawear.mbl_mw_acc_start(state.device.board)

#Opening text files
left = open(state.filename, "a")

count = 0
start_time = time.time()

try:
    while True:
        if state.data_quat and state.data_acc:
            count += 1
            current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
            print("============>Count", count, end="\r")
            left.write("[{}],[{:.3f},{:.3f},{:.3f},{:.3f}], [{:.3f},{:.3f},{:.3f}]".format(current_time , state.qw, state.qx, state.qy, state.qz, state.ax, state.ay, state.az) + "\n")
        else:
            print("No data recieved", end = "\r")
        sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    
    #left.close() #this doesn't seem to do anything? lol

    # ============= STOP =============
    libmetawear.mbl_mw_sensor_fusion_stop(state.device.board)
    libmetawear.mbl_mw_acc_stop(state.device.board)
    libmetawear.mbl_mw_acc_disable_acceleration_sampling(state.device.board)

    # ============= UNSUBSCRIBE TO SIGNAL =============
    signal_lq = libmetawear.mbl_mw_sensor_fusion_get_data_signal(state.device.board, SensorFusionData.QUATERNION)
    libmetawear.mbl_mw_datasignal_unsubscribe(signal_lq)
    signal_la = libmetawear.mbl_mw_acc_get_acceleration_data_signal(state.device.board)
    libmetawear.mbl_mw_datasignal_unsubscribe(signal_la)

    # ============= DISCONNECT =============
    libmetawear.mbl_mw_debug_disconnect(state.device.board)

    print("Total Samples Received")
    print(f"{state.device.address} -> {state.samples}")
