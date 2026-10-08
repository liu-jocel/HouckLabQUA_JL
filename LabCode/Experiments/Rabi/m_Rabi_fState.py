from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q
from scipy.optimize import curve_fit
from qualang_tools.loops import from_array
from LabCode.Helpers.fit_functions import decaying_cosine, rabi_seed
from qm import SimulationConfig
from qualang_tools.loops import from_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class RabiChevronF(ExperimentClass):
    """
    Rabi Linecut wit single shot readout
    """
    def __init__(self,element, amp_vec, f_vec, shots, outer_folder='', path='Rabi', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element #meta dict element name
        self.amp_vec_abs = amp_vec
        self.f_vec = f_vec
        self.amp_norm = self.meta_dict['pi_amp_2']
        self.amp_vec = self.amp_vec_abs/self.amp_norm
        self.freq_01 = self.meta_dict[element]['q_IF']
        self.freq_12 = self.meta_dict[element]['f_state_IF']
        self.shots = shots
        self.read_len = self.meta_dict['global_params']['read_len']

        self.data = {
            'qubit': self.element,
            "shots": self.shots,
            "amp_vec": self.amp_vec,
            "meta_dict": self.meta_dict
        }
        self.config = config
        self.job = self.acquire(self.make_prog(), self.config)
        self.analyze()
        self.data.update({'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):

        qubit = self.meta_dict['q_name']
        reset_time = self.meta_dict['reset_time']
        shots = self.shots
        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            a = declare(fixed)
            df = declare(int)
            I_stream = declare_stream()
            Q_stream = declare_stream()
            with for_(n, 0, n < shots, n + 1):
                with for_each_(df, self.f_vec):
                    with for_(*from_array(a, self.amp_vec)):
                        wait(reset_time * u.ns, qubit)
                        play('X180', qubit)
                        update_frequency(qubit, self.freq_12)
                        play('X180_2'*amp(a), qubit)
                        update_frequency(qubit, self.freq_01+df)
                        I, Q = measure_1q(self.meta_dict, I, Q)
                        save(I, I_stream)
                        save(Q, Q_stream)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")
        return prog

    def analyze(self):
        IQ_array = super().job_to_IQ_1q(self.job, [self.shots, len(self.f_vec), len(self.amp_vec)])
        I_array = np.mean(IQ_array[0], 0)
        Q_array = np.mean(IQ_array[1], 0)
        self.data.update({
            'I': I_array,
            'Q': Q_array,
            'raw_IQ': IQ_array,
        })
        plt.suptitle(
            f"{self.element} f_state_rabi ({(self.meta_dict['q_LO']['LO_freq'] + self.meta_dict['f_state_IF']) / u.MHz} MHz ).")
        plt.subplot(121)
        plt.cla()
        plt.pcolormesh(self.amp_vec_abs, self.f_vec / u.MHz, I_array, shading='nearest')
        plt.xlabel("f state pulse amplitude [V]")
        plt.ylabel("f state detuning [MHz]")
        plt.colorbar()
        plt.title("I Data")
        plt.subplot(122)
        plt.cla()
        plt.pcolormesh(self.amp_vec_abs, self.f_vec / u.MHz, Q_array, shading='nearest')
        plt.xlabel("f state pulse amplitude [V]")
        plt.ylabel("f state detuning [MHz]")
        plt.title("Q Data")
        plt.tight_layout()
        plt.colorbar()
        plt.savefig(self.iname, bbox_inches="tight")
        plt.close()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])