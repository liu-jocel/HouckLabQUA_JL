from qm.qua import *
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

matplotlib.use('Agg')
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import  measure_1q_amp
from qualang_tools.loops import from_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class TransmissionSweep(ExperimentClass):
    """
    Transmission vs readout amplitude
    """
    def __init__(self,element, amp_vec, f_vec, shots, outer_folder='', path='Rabi', suffix='data', meta_dict=None, config=None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element  # meta dict element name
        self.amp_vec = amp_vec
        self.amp_vec_scale = amp_vec/meta_dict['read_amp']
        self.num_a = len(amp_vec)
        self.f_vec = f_vec
        self.num_f = len(self.f_vec)
        self.shots = shots
        self.meta_dict = meta_dict
        self.config = config

        self.data = {
            'qubit': self.element,
            'f_vec': self.f_vec,
            "meta_dict": self.meta_dict
        }

        self.job = self.acquire(self.make_prog(), config=self.config)
        self.analyze()
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def make_prog(self):
        resonator = self.meta_dict['r_name']
        with program() as prog:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            df = declare(int)
            a = declare(fixed)
            I_stream = declare_stream()
            Q_stream = declare_stream()
            with for_(n, 0, n < self.shots, n + 1):
                with for_each_(df, self.f_vec):
                    update_frequency(resonator, df)
                    with for_(*from_array(a, self.amp_vec_scale)):
                        I, Q = measure_1q_amp(self.meta_dict, a, I, Q)
                        save(I, I_stream)
                        save(Q, Q_stream)
                        wait(self.meta_dict['resonator_dump_time'] * u.ns, resonator)

            with stream_processing():
                I_stream.save_all("I")
                Q_stream.save_all("Q")
        return prog

    def analyze(self):
        IQ_data = super().job_to_IQ_avg_1q(self.job, [self.shots, len(self.f_vec), len(self.amp_vec)])
        IQ_signal = IQ_data[0] + 1j * IQ_data[1]
        averaged_IQ_signal = np.mean(IQ_signal, 0) # dimensions (numfreqs, numamps)


        fig, ax = plt.subplots(2, 1, figsize=(10,10))
        im0 = ax[0].imshow(np.abs(averaged_IQ_signal.T), extent=[self.f_vec[0]/1e6,self.f_vec[-1]/1e6,self.amp_vec[0],self.amp_vec[-1]], interpolation='none', aspect='auto')
        ax[0].set_xlabel('IF frequency (MHz)')
        ax[0].set_ylabel(r'Readout amplitude')
        ax[0].set_title("Magnitude")
        fig.colorbar(im0, ax=ax[0])

        im1 = ax[1].imshow(np.angle(averaged_IQ_signal.T), extent=[self.f_vec[0]/1e6,self.f_vec[-1]/1e6,self.amp_vec[0],self.amp_vec[-1]], interpolation='none', aspect='auto')
        ax[1].set_xlabel('IF frequency (MHz)')
        ax[1].set_ylabel('Readout amplitude')
        ax[1].set_title("Phase")
        fig.colorbar(im1, ax=ax[1])

        read_amp = self.meta_dict['read_amp']
       #ax[0].set_title(self.meta_dict['r_name']+f' Transmission, F = {self.min_f/1e6:.2f}, Read amp: {read_amp}')
        plt.tight_layout()
        plt.savefig(self.iname, bbox_inches="tight")
        plt.close(fig)
        # offset = 0  # Initialize offset
        # colors = cm.Blues(np.linspace(0.3, 1, len(self.amp_vec)))  # Get colors from a blue colormap
        # for i, amp in enumerate(self.amp_vec):
        #     # Normalize the data
        #     normalized_data = averaged_IQ.T[i] / np.min(np.abs(averaged_IQ.T[i]))
        #     # Apply log magnitude scale
        #     log_mag_data = 20 * np.log10(np.abs(normalized_data))
        #
        #     # Plot with offset
        #     plt.plot(self.f_vec/1e6, log_mag_data + offset, label=f'Amplitude: {round(amp,4)}', color=colors[i])
        #     offset -= 1  # Increase offset for the next trace
        #
        # plt.xlabel('Frequency (MHz)')
        # plt.ylabel('Normalized Amplitude + Offset')
        # plt.title('Normalized Frequency Linecuts with Offset for Different Amplitudes')
        # plt.grid(True)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()
        # plt.savefig(self.iname, bbox_inches="tight")
        # plt.close()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data['data'])