from qm.qua import *
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append('C:/Users/my/Desktop/Houck-Lab-Qua')
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)



class Transmission(ExperimentClass):
    """
    Transmission experiement
    """
    def __init__(self,element, f_vec, shots, outer_folder='',path='trans', suffix='data', meta_dict=None,
                 config=None, plot=True, save=True):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element #meta dict element name

        # assert f_step >= 0.1e6, 'Frequency step must be at least 0.1 MHz'
        self.f_vec = f_vec
        self.num_f = len(self.f_vec)
        self.shots = shots
        self.meta_dict = meta_dict
        self.config = config
        self.plot = plot
        
        self.data = {
            'qubit': self.element,
            'f_vec': self.f_vec,
            "meta_dict": self.meta_dict
        }

        self.job = self.acquire(self.make_prog(), config=self.config)
        self.analyze()
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        if save:
            self.h5_data()

    def make_prog(self):

        resonator = self.meta_dict['r_name']
        reset_time = self.meta_dict['reset_time']
        shots = self.shots

        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            f = declare(int)
            I_stream = declare_stream()
            Q_stream = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_each_(f, self.f_vec):
                    update_frequency(resonator, f)
                    # wait(reset_time * u.ns, resonator)
                    # align()
                    # play('X180', self.element)
                    align()
                    I, Q = measure_1q(self.meta_dict, I, Q)

                    save(I, I_stream)
                    save(Q, Q_stream)


            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")

        return prog
    

    def analyze(self):

        IQ_data = super().job_to_IQ_avg_1q(self.job, [self.shots, self.num_f]) # returns list [I,Q]
        IQ_signal = IQ_data[0] + 1j*IQ_data[1]
        averaged_IQ_signal = np.mean(IQ_signal, 0) # averaged signal I + 1j*Q
        IQ_mag = np.sqrt(np.real(averaged_IQ_signal)**2 + np.imag(averaged_IQ_signal)**2)
        averaged_IQ_dBm = 20 * np.log10(IQ_mag)

        self.data.update({
            "IQ_mag": averaged_IQ_signal,
        })
        self.min_f = self.f_vec[np.argmin(np.abs(averaged_IQ_signal))]
        print("Readout IF (min) is ", self.min_f/1e6, "MHz")

        if self.plot:
            fig, ax = plt.subplots(2, 1, constrained_layout=True)
            ax[0].plot(self.f_vec*1e-6, averaged_IQ_dBm)
            ax[0].set_xlabel('IF frequency (MHz)')
            ax[0].set_ylabel(r'Power (dBm)')

            ax[1].plot(self.f_vec * 1e-6, np.angle(averaged_IQ_signal))
            ax[1].set_xlabel('IF frequency (MHz)')
            ax[1].set_ylabel('Phase')
            read_amp = self.meta_dict['read_amp']
            ax[0].set_title(self.meta_dict['r_name']+f' Transmission, F = {self.min_f/1e6:.2f}, Read amp: {read_amp}')
            plt.savefig(self.iname, bbox_inches="tight")
            plt.close(fig)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])

