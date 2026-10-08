import sys
import pyvisa as visa
import numpy as np
import time
import math,ctypes,os,csv
from PIL import Image
import struct
import matplotlib.pyplot as plt

class SAN9030A:

    # Initializes session for device.
    # VISAaddress: address of device, rm: VISA resource manager
    def __init__(self, VISAaddress, rm):
        self.VISAaddress = VISAaddress 
        try: self.N9030A = rm.open_resource(VISAaddress)
        except visa.Error as ex:
            print('Couldn\'t connect to \'%s\', exiting now...' % VISAaddress)
            sys.exit()

    #==========================================================================#

    def get_startfreq(self):
        return self.N9030A.query_ascii_values(r'FREQ:STAR?')[0]

    def get_stopfreq(self):
        return self.N9030A.query_ascii_values(r'FREQ:STOP?')[0]

    def set_cent_freq_Hz(self, cent=1000):
        cmd = r'FREQ:CENT %fHz'
        self.N9030A.write(cmd % cent)
        self.N9030A.query_ascii_values(r'*OPC?')

    def set_span_MHz(self, band=50, unit='MHz'):
        cmd = r'FREQ:SPAN %f' + unit
        self.N9030A.write(cmd % band)
        self.N9030A.write('CALC:MARK1:FUNC:BAND:SPAN %f %s' % (band, unit))
        self.N9030A.query_ascii_values(r'*OPC?')


    def set_freq_span_MHz(self, freq=1000, band=50):
        self.set_cent_freq_Hz(freq * 1e6)
        self.set_span_MHz(band)
        self.N9030A.query_ascii_values(r'*OPC?')

    def save_screen_png(self, filename):
        self.N9030A.write(r'MMEM:STOR:SCR "D:\\Temp.png"')
        binImage = self.N9030A.query_binary_values(r'MMEM:DATA? "D:\\Temp.png"', datatype='B', container=bytearray)
        path = filename + ".png"
        with open(path, 'wb') as f:
            f.write(binImage)
        f.close()

    def get_points(self):
        return int(self.N9030A.query(r'SENSE:SWEEP:POINTS?'))

    def set_points(self,points):
        try:
            self.points = int(points)
        except ValueError:
            print("points is not an integer!!!")
            print("Setting 1001 points as default")
            self.points = 1001
        self.N9030A.write(':SWE:POIN {0}'.format(self.points))

    def get_trace(self):
        ## Set analyzer to single sweep mode
        self.N9030A.write(r'INIT:CONT 0')
        ## Trigger sweep and wait for the sweep to complete
        ## self.N9030A.write(r'INIT:IMM;*WAI')
        ## Specify units in dBm
        self.N9030A.write(r'UNIT:POW DBM')
        ## Set analyzer trace data format to ASCII format
        self.N9030A.write(r'FORM:DATA ASC')
        ## Trigger sweep and wait for the sweep to complete
        self.N9030A.write(r'INIT:IMM;*WAI')
        ## Query the trace data using ascii format
        trace = self.N9030A.query_ascii_values(r'TRAC:DATA? TRACE1')
        return(trace)

def main():
    rm = visa.ResourceManager()
    VISAaddress = 'GPIB1::10::INSTR'
    sa = SAN9030A(VISAaddress, rm)
    print("Successfully connected to N9030A.")
    # LO = 3000
    # delta_F = 20
    # span = 2 * delta_F
    #
    # sa.set_freq_span_MHz(freq=LO, band=span)
    #
    # startf = sa.get_startfreq()
    # stopf = sa.get_stopfreq()
    # print(startf, stopf)
    #
    # sa.set_points(500)
    #
    # points = sa.get_points()
    # print(points, "points in trace.")

    data = sa.get_trace()
    max_signal_power = max(data)
    print(max_signal_power)
    sa.save_screen_png("2GHzQ1LO")
    #
    # result = sa.get_trace()
    # plt.plot(result)
    # plt.savefig('Test1_method2.png')

if __name__ == "__main__":
    main()

