from labinstruments.agilent import AgilentE8257D

def set_LO_parameters(LOs):
    for LO in LOs:
        addr = LO['LO_IP']
        sig = AgilentE8257D(addr)
        sig.set_frequency(LO['LO_freq'], 'Hz')
        sig.set_power(LO['LO_pow'], 'dBm')
        sig.power_on()
        sig.close()

def LO_off(LOs):
    for LO in LOs:
        sig = AgilentE8257D(LO['LO_IP'])
        sig.power_off()
        sig.close()