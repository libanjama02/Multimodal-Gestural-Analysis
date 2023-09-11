# usage: python3 stream_quat.py [mac1] [mac2] ... [mac(n)]
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


if sys.version_info[0] == 2:
    range = xrange

class State:
    # init
    def __init__(self, device="", filename=""):
        self.device = device
        self.filename = filename
        self.samples = 0
        self.callback_all = FnVoid_VoidP_DataP(self.data_handler_all)
        self.data_quat = None
        self.data_acc = None
        with  open(self.filename, "w") as file_stream:
            file_stream.write("Time,Quaternions, Accelaration \n")
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

# argument parsing
parser = OptionParser()
parser.add_option("-l", dest="left", help="MAC address of left IMU", default="D3:99:34:4D:01:CC", metavar="LEFT")
parser.add_option("-f", dest="folder", help="folder where data will be saved", default= None, metavar="FOLDER")
(options, args) = parser.parse_args()

if options.folder is None:
    raise ValueError("No filename specified for mbient data -f")

print("--left is {}".format(options.left))
print("--folder is {}".format(options.folder))

# connect
dl = MetaWear(options.left)
dl.connect()
sl = State(device=dl, filename=options.folder)
print("Connected to L={}({})".format(dl.address, "USB" if dl.usb.is_connected else "BLE"))

libmetawear.mbl_mw_settings_set_connection_parameters(sl.device.board, 7.5, 7.5, 0, 6000)
sleep(1.5)

# ============= SETUP ============= 
libmetawear.mbl_mw_sensor_fusion_set_mode(sl.device.board, SensorFusionMode.NDOF);
libmetawear.mbl_mw_sensor_fusion_set_acc_range(sl.device.board, SensorFusionAccRange._8G)
libmetawear.mbl_mw_sensor_fusion_set_gyro_range(sl.device.board, SensorFusionGyroRange._2000DPS)
libmetawear.mbl_mw_sensor_fusion_write_config(sl.device.board)
libmetawear.mbl_mw_acc_set_odr(sl.device.board, 100.0)
libmetawear.mbl_mw_acc_set_range(sl.device.board, 16.0)
libmetawear.mbl_mw_acc_write_acceleration_config(sl.device.board)

# get quat signal and subscribe
signal_lq = libmetawear.mbl_mw_sensor_fusion_get_data_signal(sl.device.board, SensorFusionData.QUATERNION)
libmetawear.mbl_mw_datasignal_subscribe(signal_lq, None, sl.callback_all)

# Get accelerometer data
signal_la = libmetawear.mbl_mw_acc_get_acceleration_data_signal(sl.device.board)
libmetawear.mbl_mw_datasignal_subscribe(signal_la, None, sl.callback_all)

# ============= START =============
libmetawear.mbl_mw_sensor_fusion_enable_data(sl.device.board, SensorFusionData.QUATERNION)
libmetawear.mbl_mw_sensor_fusion_start(sl.device.board)
libmetawear.mbl_mw_acc_enable_acceleration_sampling(sl.device.board)
libmetawear.mbl_mw_acc_start(sl.device.board)

#Opening text files
left = open(sl.filename, "a")

count = 0
start_time = time.time()

try:
    while True:
        if sl.data_quat and sl.data_acc:
            count += 1
            current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
            print("============>Count", count, end="\r")
            left.write("[{}],[{:.3f},{:.3f},{:.3f},{:.3f}], [{:.3f},{:.3f},{:.3f}]".format(current_time , sl.qw, sl.qx, sl.qy, sl.qz, sl.ax, sl.ay, sl.az) + "\n")
        else:
            print("Not data recieved", end = "\r")
        sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    left.close()

    # ============= STOP =============
    libmetawear.mbl_mw_sensor_fusion_stop(sl.device.board)
    libmetawear.mbl_mw_acc_stop(sl.device.board)
    libmetawear.mbl_mw_acc_disable_acceleration_sampling(sl.device.board)

    # ============= UNSUBSCRIBE TO SIGNAL =============
    signal_lq = libmetawear.mbl_mw_sensor_fusion_get_data_signal(sl.device.board, SensorFusionData.QUATERNION)
    libmetawear.mbl_mw_datasignal_unsubscribe(signal_lq)
    signal_la = libmetawear.mbl_mw_acc_get_acceleration_data_signal(sl.device.board)
    libmetawear.mbl_mw_datasignal_unsubscribe(signal_la)

    # ============= DISCONNECT =============
    libmetawear.mbl_mw_debug_disconnect(sl.device.board)

    print("Total Samples Received")
    print("%s -> %d" % (sl.device.address, sl.samples))
