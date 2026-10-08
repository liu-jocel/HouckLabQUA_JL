from qm.qua import *
import numpy as np
import datetime
import os
import pandas as pd
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
matplotlib.use('Agg')
from LabCode.CoreLib.experiment import ExperimentClass
from LabCode.Experiments.CR.m_CR_calibration import CalibrationCR
from LabCode.Config.config_helpers import meta_dict_to_config_CR
from qualang_tools.units import unit
u = unit(coerce_to_integer=True)


class PhaseCalibrationCR(ExperimentClass):
    """
    Frequency Calibration of 1Q gate
    """
    def __init__(self, p_vec, t_vec, shots, calib_params, confusion, num_pulses=1, outer_folder='', path='AmpCalCR',
                 suffix='data', cancel=False, echo=False, meta_dict=None, config=None,):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.meta_dict = meta_dict
        self.config = config
        self.shots = shots
        self.calib_params = calib_params
        self.confusion = confusion
        self.echo = echo
        self.cancel = cancel
        self.p_vec = p_vec
        self.t_vec = t_vec
        self.cancel = cancel
        self.echo = echo
        self.num_pulses = num_pulses

        self.data = {
            "number_pulses": num_pulses,
            "shots": self.shots,
            "t_vec": self.t_vec,
            "meta_dict": self.meta_dict
        }

        self.config = config
        self.run()
        self.data.update({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        self.h5_data()

    def run(self):
        try:
            datetime_now = datetime.now()
            date_string = datetime_now.strftime('%Y_%m_%d')
            os.mkdir(os.path.join(self.outerFolder, self.path, self.path + "_" + date_string, "Ham_tomo"))
        except:
            pass
        hamiltonian_parameters = {
            "phase": [],
            "IX": [],
            "IY": [],
            "IZ": [],
            "ZX": [],
            "ZY": [],
            "ZZ": [],
            "IX_error": [],
            "IY_error": [],
            "IZ_error": [],
            "ZX_error": [],
            "ZY_error": [],
            "ZZ_error": [],
        }
        meta_dict = self.meta_dict
        for phase in self.p_vec:
            meta_dict['qc']["ZX45_phase"] = phase
            print(phase)
            config = meta_dict_to_config_CR(self.meta_dict)
            exp = CalibrationCR(self.t_vec, self.shots, self.calib_params, self.confusion, meta_dict, config,
                                outer_folder=self.outerFolder, path=self.path, suffix=self.suffix,
                                cancel=self.cancel, echo=self.echo)

            interaction_rates = exp.data["ham_tomo_interaction_coeffs_MHz"]
            for key in interaction_rates.keys():
                rate = interaction_rates[key]
                hamiltonian_parameters[key].append(rate)
                # hamiltonian_parameters[f"{key}_error"].append(error)
            hamiltonian_parameters["phase"].append(phase)
            self.data.update({"hamiltonian_parameters": hamiltonian_parameters})
            print(f"PERCENT COMPLETE: {phase / np.max(self.p_vec) * 100 : 0.2f}%")
            self.plot(hamiltonian_parameters)
        return hamiltonian_parameters

    def plot(self, hamiltonian_parameters):
        for rate in ['IX', 'IY', 'IZ', 'ZX', 'ZY', 'ZZ']: #'ZX',
            # Extract the 'data' and 'error' from the tuples in each column
            data_values = hamiltonian_parameters[rate]
            # Plot with error bars
            plt.plot(hamiltonian_parameters['phase'], data_values, label=rate, marker='o', linestyle='-')

        # Adding labels and title
        plt.xlabel('Phase (radians)')  # x-axis label
        plt.ylabel('Interaction Rates (MHz)')  # y-axis label
        plt.title('ZX45 Phase Calibration')  # title of the plot

        # Show grid for better visualization
        plt.grid(True)

        # Displaying the legend
        plt.legend()
        plt.savefig(self.iname[:-4], bbox_inches='tight')
        plt.close()