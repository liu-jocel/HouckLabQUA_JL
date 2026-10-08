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


class AmpCalibrationCR(ExperimentClass):
    """
    Frequency Calibration of 1Q gate
    """
    def __init__(self, a_vec, t_vec, shots, calib_params, confusion, num_pulses=1, outer_folder='', path='AmpCalCR',
                 suffix='data', cancel=False, echo=False, meta_dict=None, config=None,):
        super().__init__(path=path, outer_folder=outer_folder,suffix=suffix, meta_dict=meta_dict, config=config)
        self.meta_dict = meta_dict
        self.config = config
        self.shots = shots
        self.calib_params = calib_params
        self.confusion = confusion
        self.echo = echo
        self.cancel = cancel
        self.a_vec = a_vec
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
            "amp": [],
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
        for a in self.a_vec:
            meta_dict['qc']["ZX45_amp"] = a
            print(a)
            config = meta_dict_to_config_CR(self.meta_dict)
            exp = CalibrationCR(self.t_vec, self.shots, self.calib_params, self.confusion, meta_dict, config,
                                outer_folder=self.outerFolder, path=self.path, suffix=self.suffix,
                                cancel=self.cancel, echo=self.echo)

            interaction_rates = exp.data["ham_tomo_interaction_coeffs_MHz"]
            for key in interaction_rates.keys():
                rate = interaction_rates[key]
                hamiltonian_parameters[key].append(rate)
                # hamiltonian_parameters[f"{key}_error"].append(error)
            hamiltonian_parameters["amp"].append(a)
            self.data.update({"hamiltonian_parameters": hamiltonian_parameters})
            print(f"PERCENT COMPLETE: {a / np.max(self.a_vec) * 100 : 0.2f}%")
            self.plot(hamiltonian_parameters)
        return hamiltonian_parameters


    def plot(self, hamiltonian_parameters):
        for rate in ['IX', 'IY', 'IZ', 'ZX', 'ZY', 'ZZ']:
            # Extract the 'data' and 'error' from the tuples in each column
            data_values = hamiltonian_parameters[rate]
            errors = hamiltonian_parameters[f"{rate}_error"]
            # Plot with error bars
            plt.plot(hamiltonian_parameters['amp'], data_values, label=rate, marker='o', linestyle='-')

        # Adding labels and title
        plt.xlabel('Amp (arb)')  # x-axis label
        plt.ylabel('Interaction Rates (MHz)')  # y-axis label
        plt.title('ZX45 Amplitude Calibration')  # title of the plot

        # Show grid for better visualization
        plt.grid(True)

        # Displaying the legend
        plt.legend()
        plt.savefig(self.iname[:-4], bbox_inches='tight')
        plt.close()