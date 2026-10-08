import sys
import visa
import numpy as np
import time


class AWG33220A:

    # Initializes session for device.
    # VISAaddress: address of device, rm: VISA resource manager
    def __init__(self, VISAaddress, rm):
        self.VISAaddress = VISAaddress
        try:
            self.session = rm.open_resource(VISAaddress)
        except visa.Error as ex:
            sys.stderr.write('Couldn\'t connect to \'%s\', exiting now...' \
                             % VISAaddress)
            sys.exit()

    # ==========================================================================#

    # Turn on output
    def PowerOn(self):
        self.session.write('OUTPut ON')

    # Turn off output
    def PowerOff(self):
        self.session.write('OUTPut OFF')

    # ==========================================================================#

    # Set waveform
    def SetWaveform(self, waveform):
        formlist = {'SIN', 'SQUARE', 'RAMP', 'PULSE', 'NOISE', 'DC', 'USER'}
        if not (waveform in formlist):
            sys.stderr.write("Error: waveform needs to be one of " + str(formlist) + "\n")
            return
        self.session.write('FUNCtion %s' % waveform)

    # Set pulse width
    def SetPulseWidth(self, pulsewidth):
        self.session.write('PULSe:WIDTh %f' % pulsewidth)

    # Set edge time
    def SetEdgeTime(self, edgetime):
        self.session.write('PULSe:TRANsition %f' % edgetime)

    # Set frequency
    def SetFreq(self, frequency):
        self.session.write('FREQuency %f' % frequency)

    # Set period
    def SetPeriod(self, period):
        self.session.write('FREQuency %f' % 1 / period)

    # Set peak to peak amplitude (voltage)
    def SetVPP(self, vpp):
        self.session.write('VOLTage %f' % vpp)

    # Set voltage offset
    def SetOffset(self, offset):
        self.session.write('VOLTage:OFFSet %f' % offset)

    # Set duty cycle for a square waveform (in percent)
    def SetDutyCycle(self, dutycycle):
        freq = self.GetFrequency()
        if freq <= 10e6 and (dutycycle < 20 or dutycycle > 80):
            sys.stderr.write("Error: Duty cycle needs to be between 20% and 80% " + \
                             "for frequency <= 10 MHz")
            return
        if freq > 10e6 and (dutycycle < 40 or dutycycle > 60):
            sys.stderr.write("Error: Duty cycle needs to be between 20% and 80% " + \
                             "for frequency > 10 MHz")
            return
        self.session.write('FUNCtion:SQUare:DCYCle %f' % dutycycle)

    # ==========================================================================#

    # Returns the waveform
    def GetWaveform(self):
        self.session.write('FUNCtion?')
        result = self.session.read()
        return result.rstrip('\n')

    # Returns the freq as a float
    def GetFreq(self):
        self.session.write('FREQuency?')
        result = self.session.read()
        return float(result.rstrip('\n'))

    def GetPulseWidth(self):
        self.session.write('PULSe:WIDTh?')
        result = self.session.read()
        return float(result.rstrip('\n'))

    def GetEdgeTime(self):
        self.session.write('PULSe:TRANsition?')
        result = self.session.read()
        return float(result.rstrip('\n'))

    # Returns the period as a float
    def GetPeriod(self):
        self.session.write('FREQuency?')
        result = self.session.read()
        return 1 / float(result.rstrip('\n'))

    # Returns the peak to peak amplitude (voltage) as a float
    def GetVPP(self):
        self.session.write('VOLTage?')
        result = self.session.read()
        return float(result.rstrip('\n'))

    # Returns the offset as a float
    def GetOffset(self):
        self.session.write('VOLTage:OFFSet?')
        result = self.session.read()
        return float(result.rstrip('\n'))

    # Returns the duty cycle as a float
    def GetDutyCycle(self):
        self.session.write('FUNCtion:SQUare:DYCYCle?')
        result = self.session.read()
        return float(result.rstrip('\n'))

    # ==========================================================================#


def main():
    rm = visa.ResourceManager()
    VISAaddress = 'USB0::0x0957::0x4B07::MY53402082::0::INSTR'
    # VISAaddress = 'TCPIP0::192.168.1.131::inst0::INSTR'
    awg = AWG33220A(VISAaddress, rm)
    awg.PowerOff()
    awg.SetWaveform('PULSE')
    print("Waveform type is =", awg.GetWaveform())
    awg.SetFreq(1000) # 125 Hz = 8 ms
    print("Frequency in Hz is=", awg.GetFreq())
    awg.SetVPP(1)
    print("Vpp in volts is =", awg.GetVPP())
    awg.SetOffset(0.5)
    print("Offset in volts is=", awg.GetOffset())

    # awg.SetEdgeTime(5e-9)
    # awg.SetPulseWidth(20e-9) # To do need to fix this to set width for 33600A
    print("Pulse width is=",awg.GetPulseWidth())
    # Edge time is minimal 2.9 ns.

    # print("Pulse edge time is=",awg.GetEdgeTime())
    # #
    # awg.SetFreq(0.000125e6)
    # print("Frequency in Hz is=", awg.GetFreq())
    # # time.sleep(2)
    # awg.SetVPP(1)
    # print("Vpp in volts is =", awg.GetVPP())
    # awg.SetOffset(0.5)
    # print("Offset in volts is=", awg.GetOffset())

    ### to do: set pulse width 8 ns
    ### to do: set edge time 5 ns

    awg.PowerOn()
    print("Finished running")

if __name__ == "__main__":
    main()

