from qm.qua import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Helpers.qua_helpers import measure_1q_amp
from LabCode.Helpers.ss_helpers import find_blob_mean, find_threshold
from qualang_tools.loops import from_array
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

class SS_Cal_1Q(ExperimentClass):
    """
    Single shot calibration experiment
    """
    def __init__(self,element, a_vec, f_vec, shots, outer_folder='',path='SingleShotCal', suffix='data', meta_dict=None, config = None):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)

        self.element = element # meta dict element name
        self.config = config
        self.shots = shots
        self.a_vec = a_vec
        self.f_vec = f_vec
        self.data={
            'meta_dict': meta_dict,
            'a_vec': self.a_vec,
            'f_vec': self.f_vec,
            'shots': self.shots,
        }

        self.job = self.acquire(self.make_prog(), self.config)
        self.max_F = self.analyze()
        self.h5_data()

    def make_prog(self):
        qubit = self.meta_dict['q_name']
        resonator = self.meta_dict['r_LO']["LO_name"]
        reset_time = self.meta_dict['reset_time']
        shots = self.shots
        res_lo_freq = self.meta_dict['r_LO']["LO_freq"]


        with program() as prog:
            n = declare(int)
            df = declare(int)
            a = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)
            I_stream_0 = declare_stream()
            Q_stream_0 = declare_stream()
            I_stream_1 = declare_stream()
            Q_stream_1 = declare_stream()

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(a, self.a_vec)):
                    with for_(*from_array(df, self.f_vec)):
                        wait(reset_time * u.ns, qubit)
                        update_frequency(resonator, df + res_lo_freq)
                        I0, Q0 = measure_1q_amp(self.meta_dict, a, I, Q)

                        save(I0, I_stream_0)
                        save(Q0, Q_stream_0)

            with for_(n, 0, n < shots, n + 1):
                with for_(*from_array(a, self.a_vec)):
                    with for_(*from_array(df, self.f_vec)):
                        wait(reset_time * u.ns, qubit)
                        play('X180', qubit)
                        play('X180', "q1")
                        update_frequency(resonator, df + res_lo_freq)
                        I1, Q1 = measure_1q_amp(self.meta_dict, a, I, Q)

                        save(I1, I_stream_1)
                        save(Q1, Q_stream_1)

            with stream_processing():
                I_stream_0.save_all("I_0")
                Q_stream_0.save_all("Q_0")
                I_stream_1.save_all("I_1")
                Q_stream_1.save_all("Q_1")

        return prog

    def analyze(self):
        res_LO_freq = self.meta_dict['r_LO']["LO_freq"]

        job = self.job

        I_handle_0 = job.result_handles.get("I_0")
        Q_handle_0 = job.result_handles.get("Q_0")
        I_handle_1 = job.result_handles.get("I_1")
        Q_handle_1 = job.result_handles.get("Q_1")

        I_handle_0.wait_for_all_values()
        Q_handle_0.wait_for_all_values()
        I_handle_1.wait_for_all_values()
        Q_handle_1.wait_for_all_values()

        shape = (self.shots, len(self.a_vec), len(self.f_vec))
        # 0shots 1amps 2freqs. need to loop over amps and freqs
        I_0 = I_handle_0.fetch_all()['value'].reshape(shape)
        Q_0 = Q_handle_0.fetch_all()['value'].reshape(shape)
        I_1 = I_handle_1.fetch_all()['value'].reshape(shape)
        Q_1 = Q_handle_1.fetch_all()['value'].reshape(shape)

        fidelities = np.zeros((len(self.a_vec), len(self.f_vec)))
        for i in range(len(self.a_vec)):
            for j in range(len(self.f_vec)):
                c_0 = I_0[:, i, j] + 1j * Q_0[:, i, j]
                c_1 = I_1[:, i, j] + 1j * Q_1[:, i, j]

                c_theta = np.angle(find_blob_mean(c_1) - find_blob_mean(c_0))

                c_0_rot = np.exp(-1j * c_theta) * c_0
                c_1_rot = np.exp(-1j * c_theta) * c_1

                c_thresh, c_F = find_threshold(c_0_rot, c_1_rot)
                fidelities[i,j] = np.max(c_F)
        
        min_fidelity = np.max([np.min([fidelities]),0.5]) # 0.5 is the lowest readout fidelity. 
        max_fidelity_index = np.unravel_index(np.argmax(fidelities), fidelities.shape)
        max_fidelity = fidelities[max_fidelity_index]
        
        self.data.update({
            'fidelities': fidelities
        })

        optimal_amp = self.a_vec[max_fidelity_index[0]]
        optimal_frequency = (self.f_vec[max_fidelity_index[1]]) / u.MHz

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        pcm = ax.pcolormesh(self.a_vec, (self.f_vec) / u.MHz, fidelities, vmin=min_fidelity, vmax=max_fidelity)
        plt.colorbar(pcm)
        ax.set_xlabel('Readout amplitude (V)')
        ax.set_ylabel('Resonator Detuning (MHz)')
        ax.set_title(f"{self.element} readout calibration. Max fidelity: {max_fidelity}, Amplitude: {optimal_amp} V, Frequency: {optimal_frequency} MHz.")
        plt.savefig(self.iname[:-4]+"_blob", bbox_inches = 'tight')
        plt.close(fig)