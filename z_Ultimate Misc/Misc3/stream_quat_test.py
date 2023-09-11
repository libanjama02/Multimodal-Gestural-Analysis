from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
from time import sleep

class State:
    def __init__(self, device):
        self.device = device
        self.samples = 0
        self.callback = FnVoid_VoidP_DataP(self.data_handler)

    def data_handler(self, ctx, data):
        print("QUAT: %s -> %s" % (self.device.address, parse_value(data)))
        self.samples += 1

# Hardcoded MAC address
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

libmetawear.mbl_mw_sensor_fusion_enable_data(state.device.board, SensorFusionData.QUATERNION)
libmetawear.mbl_mw_sensor_fusion_start(state.device.board)

sleep(10.0)

libmetawear.mbl_mw_sensor_fusion_stop(state.device.board)
libmetawear.mbl_mw_datasignal_unsubscribe(signal)
libmetawear.mbl_mw_debug_disconnect(state.device.board)

print("Total Samples Received: %d" % state.samples)

