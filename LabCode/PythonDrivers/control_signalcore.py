import ctypes
from typing import Dict
from qcodes import Instrument
from qcodes.utils.validators import Enum
import sys

MAXDEVICES = 50
MAXDESCRIPTORSIZE = 9
COMMINTERFACE = ctypes.c_uint8(1)

class ManDate(ctypes.Structure):
    _fields_ = [('year', ctypes.c_uint8),
                ('month', ctypes.c_uint8),
                ('day', ctypes.c_uint8),
                ('hour', ctypes.c_uint8)]


class DeviceInfoT(ctypes.Structure):
    _fields_ = [('product_serial_number', ctypes.c_uint32),
                ('hardware_revision', ctypes.c_float),
                ('firmware_revision', ctypes.c_float),
                ('device_interfaces', ctypes.c_uint8),
                ('man_date', ManDate)]
device_info_t = DeviceInfoT()


class ListModeT(ctypes.Structure):
    _fields_ = [('sweep_mode', ctypes.c_uint8),
                ('sweep_dir', ctypes.c_uint8),
                ('tri_waveform', ctypes.c_uint8),
                ('hw_trigger', ctypes.c_uint8),
                ('step_on_hw_trig', ctypes.c_uint8),
                ('return_to_start', ctypes.c_uint8),
                ('trig_out_enable', ctypes.c_uint8),
                ('trig_out_on_cycle', ctypes.c_uint8)]


class PLLStatusT(ctypes.Structure):
    _fields_ = [('sum_pll_ld', ctypes.c_uint8),
                ('crs_pll_ld', ctypes.c_uint8),
                ('fine_pll_ld', ctypes.c_uint8),
                ('crs_ref_pll_ld', ctypes.c_uint8),
                ('crs_aux_pll_ld', ctypes.c_uint8),
                ('ref_100_pll_ld', ctypes.c_uint8),
                ('ref_10_pll_ld', ctypes.c_uint8)]


class OperateStatusT(ctypes.Structure):
    _fields_ = [('rf1_lock_mode', ctypes.c_uint8),
                ('rf1_loop_gain', ctypes.c_uint8),
                ('device_access', ctypes.c_uint8),
                ('rf2_standby', ctypes.c_uint8),
                ('rf1_standby', ctypes.c_uint8),
                ('auto_pwr_disable', ctypes.c_uint8),
                ('alc_mode', ctypes.c_uint8),
                ('rf1_out_enable', ctypes.c_uint8),
                ('ext_ref_lock_enable', ctypes.c_uint8),
                ('ext_ref_detect', ctypes.c_uint8),
                ('ref_out_select', ctypes.c_uint8),
                ('list_mode_running', ctypes.c_uint8),
                ('rf1_mode', ctypes.c_uint8),
                ('over_temp', ctypes.c_uint8),
                ('pxi_clk_enable', ctypes.c_uint8),
                ('harmonic_ss', ctypes.c_uint8)]

class DeviceStatusT(ctypes.Structure):
    _fields_ = [('list_mode_t', ListModeT),
                ('operate_status_t', OperateStatusT),
                ('pll_status_t', PLLStatusT)]
device_status_t = DeviceStatusT()


class HWTriggerT(ctypes.Structure):
    _fields_ = [('edge', ctypes.c_uint8),
                ('pxi_enable', ctypes.c_uint8),
                ('pxi_line', ctypes.c_uint8)]
hw_trigger_t = HWTriggerT()


class DeviceRFParamsT(ctypes.Structure):
    _fields_ = [('frequency', ctypes.c_ulonglong),
                ('sweep_start_freq', ctypes.c_ulonglong),
                ('sweep_stop_freq', ctypes.c_ulonglong),
                ('sweep_step_freq', ctypes.c_ulonglong),
                ('sweep_dwell_time', ctypes.c_ulong),
                ('sweep_cycles', ctypes.c_ulong),
                ('buffer_points', ctypes.c_ulong),
                ('power_level', ctypes.c_float),
                ('frequency_rf2', ctypes.c_ushort)]

device_rf_params_t = DeviceRFParamsT()


error_dict = {'0':'SCI_SUCCESS',
              '0':'SCI_ERROR_NONE',
              '-1':'SCI_ERROR_INVALID_DEVICE_HANDLE',
              '-2':'SCI_ERROR_NO_DEVICE',
              '-3':'SCI_ERROR_INVALID_DEVICE',
              '-4':'SCI_ERROR_MEM_UNALLOCATE',
              '-5':'SCI_ERROR_MEM_EXCEEDED',
              '-6':'SCI_ERROR_INVALID_REG',
              '-7':'SCI_ERROR_INVALID_ARGUMENT',
              '-8':'SCI_ERROR_COMM_FAIL',
              '-9':'SCI_ERROR_OUT_OF_RANGE',
              '-10':'SCI_ERROR_PLL_LOCK',
              '-11':'SCI_ERROR_TIMED_OUT',
              '-12':'SCI_ERROR_COMM_INIT',
              '-13':'SCI_ERROR_TIMED_OUT_READ',
              '-14':'SCI_ERROR_INVALID_INTERFACE'}



class SC5510A(Instrument):
    __doc__ = 'QCoDeS python driver for the Signal Core sc5510a.'

    def __init__(self, num: int,
                       dll_path: str='C:\\Program Files\\SignalCore\\SC5510A\\api\\scipci\\c\\lib\\x64\\sc5510a.dll',
                       **kwargs):
        """
        QCoDeS driver for the Signal Core sc5510a (managed by Sara Sussman sarafs@princeton.edu)

        Args:
        num (int): Number of the instrument (31-36)
        dll_path (str): Path towards the instrument DLL.
        """
        
        (super().__init__)(num, **kwargs)
        self._dll = ctypes.WinDLL(dll_path)
        self._devices_number = ctypes.c_uint()
        self._pxi10Enable = 0
        self._lock_external = 0
        self._clock_frequency = 10
        buffers = [ctypes.create_string_buffer(MAXDESCRIPTORSIZE + 1) for bid in range(MAXDEVICES)]
        self.buffer_pointer_array = (ctypes.c_char_p * MAXDEVICES)()
        for device in range(MAXDEVICES):
            self.buffer_pointer_array[device] = ctypes.cast(buffers[device], ctypes.c_char_p)

        self._buffer_pointer_array_p = ctypes.cast(self.buffer_pointer_array, ctypes.POINTER(ctypes.c_char_p))
        # sc5510a_search_devices searches for SignalCore SC5510A devices available on the host
        # computer and returns the size (number) of devices found and populates the char array
        # with their visa_device resources. The user can use this information to open specific
        # device(s) based on the visa_device resource. See sc5510a_open_device function on how
        # to open a device.
        found = self._dll.sc5510a_search_devices(self._buffer_pointer_array_p, ctypes.byref(self._devices_number))
        if found:
            raise RuntimeError('Failed to find any device')

        num_of_devices = self._devices_number.value
        print("There are %d SC5510A devices found. Accessing device with label %d" % (num_of_devices,num))
        list_of_devices = []
        for i in range(0, num_of_devices):
            #print("%d. %s" % (i + 1, self._buffer_pointer_array_p[i].decode("utf-8")[-2:]))
            list_of_devices.append(int(self._buffer_pointer_array_p[i].decode("utf-8")[-2:]))
        #print(list_of_devices)
        index_to_open = list_of_devices.index(num)
        #print(index_to_open)
        self._open(index_to_open)

        self.add_parameter(name='serialnum',
                           docstring='Return the device serial number.',
                           get_cmd=self._get_serial)

        self.add_parameter(name='temperature',
                           docstring='Return the microwave source internal temperature.',
                           label='Device temperature',
                           unit='celsius',
                           get_cmd=self._get_temperature)
        
        self.add_parameter(name='output',
                           docstring='.',
                           set_cmd=self._set_status,
                           get_cmd=self._get_status)
        
        self.add_parameter(name='power',
                           docstring='.',
                           label='Power',
                           unit='dbm',
                           set_cmd=self._set_power,
                           get_cmd=self._get_power)
        
        self.add_parameter(name='frequency',
                           docstring='.',
                           label='Frequency',
                           unit='Hz',
                           set_cmd=self._set_frequency,
                           get_cmd=self._get_frequency)
        self.add_parameter(name='rf_mode',
                           docstring='.',
                           vals=Enum('single_tone', 'sweep'),
                           #initial_value='single_tone',
                           set_cmd=self._set_rf_mode,
                           get_cmd=self._get_rf_mode)
        
        self.add_parameter(name='clock_reference',
                           docstring='Select the clock reference, internal or external.',
                           vals=Enum('internal', 'external'),
                           #initial_value='internal',
                           set_cmd=self._set_clock_reference,
                           get_cmd=self._get_clock_reference)

    def _open(self,device_num) -> None:
        # sc5510a_open_device opens the device and turns the front panel “active” LED on if it is
        # successful. It returns a handle to the device for other function calls.
        self._handle = ctypes.wintypes.HANDLE()
        self._dll.sc5510a_open_device(self.buffer_pointer_array[device_num], ctypes.byref(self._handle))

    def _close(self) -> None:
        # sc5510a_close_device closes the device associated with the device handle and turns off
        # the “active” LED on the front panel if it is successful
        self._dll.sc5510a_close_device(self._handle)

    def _get_temperature(self) -> float:
        temperature = ctypes.c_float()
        self._dll.sc5510a_get_temperature(self._handle, ctypes.byref(temperature))
        return temperature.value

    def _set_status(self, status: bool) -> None:
        # sc5510a_set_output enables or disable the output RF1.
        if status == True:
            status_ = 1
        else:
            status_ = 0
        self._dll.sc5510a_set_output(self._handle, ctypes.c_int(status_))

    def _get_status(self) -> bool:
        # sc5510a_get_device_status gets the current device status such as the PLL lock status,
        # sweep modes, and other operating conditions.
        self._dll.sc5510a_get_device_status(self._handle, ctypes.byref(device_status_t))
        return bool(device_status_t.operate_status_t.rf1_out_enable)

    def _set_power(self, power: float) -> None:
        # sc5510a_set_power_level set the power output level of RF1.
        self._dll.sc5510a_set_level(self._handle, ctypes.c_float(power))

    def _get_power(self) -> float:
        # sc5510a_get_rf_parameters gets the current RF parameters such as RF1 frequency, RF2
        # frequency, and sweep start frequency, etc.
        self._dll.sc5510a_get_rf_parameters(self._handle, ctypes.byref(device_rf_params_t))
        return device_rf_params_t.power_level

    def _set_frequency(self, frequency: float) -> None:
        # sc5510a_set_frequency sets RF1 frequency.
        msg = self._dll.sc5510a_set_freq(self._handle, ctypes.c_ulonglong(int(frequency)))

    def _get_frequency(self) -> float:
        # sc5510a_get_rf_parameters gets the current RF parameters such as RF1 frequency, RF2
        # frequency, and sweep start frequency, etc.
        device_rf_params_t = DeviceRFParamsT()
        self._dll.sc5510a_get_rf_parameters(self._handle, ctypes.byref(device_rf_params_t))
        return float(device_rf_params_t.frequency)

    def _get_clock_frequency(self) -> float:
        # sc5510a_get_device_status gets the current device status such as the PLL lock status,
        # sweep modes, and other operating conditions.
        self._dll.sc5510a_get_device_status(self._handle, ctypes.byref(device_status_t))
        ref_out_select = device_status_t.operate_status_t.ref_out_select
        if ref_out_select == 1:
            return 100
        return 10

    def _set_clock_reference(self, clock_reference: str) -> None:
        # sc5510a_set_clock_reference configure the reference clock (ref out) behavior.
        if clock_reference.lower() == 'internal':
            self._lock_external = 0
            print("Setting to internal now")
        else:
            self._lock_external = 1
        self._select_high = 0
        self._dll.sc5510a_set_clock_reference(self._handle, ctypes.c_int(self._pxi10Enable), ctypes.c_int(self._select_high), ctypes.c_int(self._lock_external))

    def _get_clock_reference(self) -> str:
        # sc5510a_get_device_status gets the current device status such as the PLL lock status,
        # sweep modes, and other operating conditions.
        self._dll.sc5510a_get_device_status(self._handle, ctypes.byref(device_status_t))
        ext_ref_detect = device_status_t.operate_status_t.ext_ref_detect
        if ext_ref_detect == 1:
            return 'external'
        return 'internal'

    def _set_rf_mode(self, rf_mode: str) -> None:
        # sc5510a_set_rf_mode sets RF1 to fixed tone or sweep.
        if rf_mode.lower() == 'single_tone':
            self.rf_mode_ = 0
        else:
            self.rf_mode_ = 1
        self._dll.sc5510a_set_rf_mode(self._handle, ctypes.c_int(self.rf_mode_))

    def _get_rf_mode(self) -> str:
        # sc5510a_get_device_status gets the current device status such as the PLL lock status,
        # sweep modes, and other operating conditions.
        self._dll.sc5510a_get_device_status(self._handle, ctypes.byref(device_status_t))
        rf_mode = device_status_t.operate_status_t.rf1_mode
        if rf_mode == 0:
            return 'single_tone'
        return 'sweep'

    def _get_serial(self) -> Dict[(str, str)]:
        # sc5510a_get_device_info obtains the device information such as serial number,
        # hardware revision, firmware revision, and manufactured date.
        self._dll.sc5510a_get_device_info(self._handle, ctypes.byref(device_info_t))
        return device_info_t.product_serial_number

def setLO(module, out, rf_power, freq):
    sc = SC5510A(module)
    print("\n Getting parameters...")
    sc.serialnum()
    sc.clock_reference()
    sc.frequency()
    sc.power()
    sc.rf_mode()
    sc.output()
    sc.temperature()
    sc.print_readable_snapshot()
    print("\n Setting parameters...")
    sc.output(out)
    sc.power(rf_power)
    sc.frequency(freq)
    print("\n Getting parameters...")
    sc.serialnum()
    sc.clock_reference()
    sc.frequency()
    sc.power()
    sc.rf_mode()
    sc.output()
    sc.temperature()
    sc.print_readable_snapshot()
    return sc

def create_LOs(LO_list):
    LO_objs = {}

    for lo in LO_list:
        LO = setLO(lo['LO_mod'],1,lo['LO_pow'],lo['LO_freq'])
        LO_objs.update({lo['LO_name']:LO})

    return LO_objs

class SignalCoreLOs:

    def __init__(self,LO_list):
        self.LOs = create_LOs(LO_list)

    def get_freq(self,LO):
        self.LOs[LO].frequncy()

    def set_freq(self,LO,f):
        self.LOs[LO].frequncy(f)

    def shutdown(self):
        for key in self.LOs.keys():
            self.LOs[key]._close()

