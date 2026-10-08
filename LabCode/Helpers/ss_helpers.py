import numpy as np

def convert_e_population_to_expectation(pop):
    return 1-2*np.array(pop)



class Discrim2QArb:
    def __init__(self, POVM_dict):

        self.POVM_dict = POVM_dict
        self.calib_params = self.make_calib_params()
        self.confusion = self.get_confusion()
        self.readout_fidelity = self.get_readout_fidelity()

    def make_calib_params(self):
        pass

    def get_confusion(self):
        pass

    def get_readout_fidelity(self):
        pass

    def get_state(self):
        pass

class Discrim2Q2L:

    def __init__(self, POVM_dict, POVM):

        self.POVM_dict = POVM_dict
        self.POVM = POVM
        self.calib_params = self.make_calib_params()
        self.confusion = self.get_confusion()
        self.readout_fidelity = self.get_readout_fidelity()

    def make_calib_params(self):

        IQ_00 = self.POVM_dict['00']
        IQ_01 = self.POVM_dict['01']
        IQ_10 = self.POVM_dict['10']
        IQ_11 = self.POVM_dict['11']

        I1_00 = np.real(IQ_00['IQ1'])
        Q1_00 = np.imag(IQ_00['IQ1'])
        I1_01 = np.real(IQ_01['IQ1'])
        Q1_01 = np.imag(IQ_01['IQ1'])
        I1_10 = np.real(IQ_10['IQ1'])
        Q1_10 = np.imag(IQ_10['IQ1'])
        I1_11 = np.real(IQ_11['IQ1'])
        Q1_11 = np.imag(IQ_11['IQ1'])

        I2_00 = np.real(IQ_00['IQ2'])
        Q2_00 = np.imag(IQ_00['IQ2'])
        I2_01 = np.real(IQ_01['IQ2'])
        Q2_01 = np.imag(IQ_01['IQ2'])
        I2_10 = np.real(IQ_10['IQ2'])
        Q2_10 = np.imag(IQ_10['IQ2'])
        I2_11 = np.real(IQ_11['IQ2'])
        Q2_11 = np.imag(IQ_11['IQ2'])

        c1_0_mean = [np.mean([I1_00, I1_01]), np.mean([Q1_00, Q1_01])]
        c1_1_mean = [np.mean([I1_10, I1_11]), np.mean([Q1_10, Q1_11])]
        c2_0_mean = [np.mean([I2_00, I2_10]), np.mean([Q2_00, Q2_10])]
        c2_1_mean = [np.mean([I2_01, I2_11]), np.mean([Q2_01, Q2_11])]

        calib_params = {'c1_0_mean': c1_0_mean,
                        'c1_1_mean': c1_1_mean,
                        'c2_0_mean': c2_0_mean,
                        'c2_1_mean': c2_1_mean}

        return calib_params

    def string_to_index(self,state):
        index = 0
        for digit in state:
            index = index * 2 + int(digit)
        return index

    def get_confusion(self):


        IQ_00 = self.POVM_dict['00']
        IQ_01 = self.POVM_dict['01']
        IQ_10 = self.POVM_dict['10']
        IQ_11 = self.POVM_dict['11']
        num_shots = len(IQ_00['IQ1'])
        confusion = np.zeros([4, 4])
        IQ_dict = {
                    'IQ_00': IQ_00,
                    'IQ_01': IQ_01,
                    'IQ_10': IQ_10,
                    'IQ_11': IQ_11
                    }
        for i in range(num_shots):
            for k,key in enumerate(IQ_dict.keys()):
                IQ_point = {'IQ1': IQ_dict[key]['IQ1'][i],
                            'IQ2': IQ_dict[key]['IQ2'][i]
                            }

                state_string = self.get_state(IQ_point)
                state_index = self.string_to_index(state_string)
                confusion[state_index][k] += 1 / num_shots
        return confusion

    def get_readout_fidelity(self):

        return np.trace(self.confusion)/4

    def get_state(self, IQ_point):
        c1_0_mean = self.calib_params['c1_0_mean']
        c1_1_mean = self.calib_params['c1_1_mean']
        c2_0_mean = self.calib_params['c2_0_mean']
        c2_1_mean = self.calib_params['c2_1_mean']

        s_q1 = find_state([np.real(IQ_point['IQ1']),np.imag(IQ_point['IQ1'])], c1_0_mean, c1_1_mean)
        s_q2 = find_state([np.real(IQ_point['IQ2']),np.imag(IQ_point['IQ2'])], c2_0_mean, c2_1_mean)

        return f"{s_q1}{s_q2}"


class Discrim2Q3L:

    def __init__(self, POVM_dict, POVM):

        self.POVM_dict = POVM_dict
        self.POVM = POVM
        self.calib_params = self.make_calib_params()
        self.confusion = self.get_confusion()
        self.readout_fidelity = self.get_readout_fidelity()

    def make_calib_params(self):

        IQ_00 = self.POVM_dict['00']
        IQ_01 = self.POVM_dict['01']
        IQ_02 = self.POVM_dict['02']
        IQ_10 = self.POVM_dict['10']
        IQ_11 = self.POVM_dict['11']
        IQ_12 = self.POVM_dict['12']
        IQ_20 = self.POVM_dict['20']
        IQ_21 = self.POVM_dict['21']
        IQ_22 = self.POVM_dict['22']


        I1_00 = np.real(IQ_00['IQ1'])
        Q1_00 = np.imag(IQ_00['IQ1'])
        I1_01 = np.real(IQ_01['IQ1'])
        Q1_01 = np.imag(IQ_01['IQ1'])
        I1_02 = np.real(IQ_02['IQ1'])
        Q1_02 = np.imag(IQ_02['IQ1'])
        I1_10 = np.real(IQ_10['IQ1'])
        Q1_10 = np.imag(IQ_10['IQ1'])
        I1_11 = np.real(IQ_11['IQ1'])
        Q1_11 = np.imag(IQ_11['IQ1'])
        I1_12 = np.real(IQ_12['IQ1'])
        Q1_12 = np.imag(IQ_12['IQ1'])
        I1_20 = np.real(IQ_20['IQ1'])
        Q1_20 = np.imag(IQ_20['IQ1'])
        I1_21 = np.real(IQ_21['IQ1'])
        Q1_21 = np.imag(IQ_21['IQ1'])
        I1_22 = np.real(IQ_22['IQ1'])
        Q1_22 = np.imag(IQ_22['IQ1'])




        I2_00 = np.real(IQ_00['IQ2'])
        Q2_00 = np.imag(IQ_00['IQ2'])
        I2_01 = np.real(IQ_01['IQ2'])
        Q2_01 = np.imag(IQ_01['IQ2'])
        I2_02 = np.real(IQ_02['IQ2'])
        Q2_02 = np.imag(IQ_02['IQ2'])
        I2_10 = np.real(IQ_10['IQ2'])
        Q2_10 = np.imag(IQ_10['IQ2'])
        I2_11 = np.real(IQ_11['IQ2'])
        Q2_11 = np.imag(IQ_11['IQ2'])
        I2_12 = np.real(IQ_12['IQ2'])
        Q2_12 = np.imag(IQ_12['IQ2'])
        I2_20 = np.real(IQ_20['IQ2'])
        Q2_20 = np.imag(IQ_20['IQ2'])
        I2_21 = np.real(IQ_21['IQ2'])
        Q2_21 = np.imag(IQ_21['IQ2'])
        I2_22 = np.real(IQ_22['IQ2'])
        Q2_22 = np.imag(IQ_22['IQ2'])

        c1_0_mean = [np.mean([I1_00, I1_01, I1_02]), np.mean([Q1_00, Q1_01, Q1_02])]
        c1_1_mean = [np.mean([I1_10, I1_11, I1_12]), np.mean([Q1_10, Q1_11, Q1_12])]
        c1_2_mean = [np.mean([I1_20, I1_21, I1_22]), np.mean([Q1_20, Q1_21, Q1_22])]
        c2_0_mean = [np.mean([I2_00, I2_10, I2_20]), np.mean([Q2_00, Q2_10, Q2_20])]
        c2_1_mean = [np.mean([I2_01, I2_11, I2_21]), np.mean([Q2_01, Q2_11, Q2_21])]
        c2_2_mean = [np.mean([I2_02, I2_12, I2_22]), np.mean([Q2_02, Q2_12, Q2_22])]

        calib_params = {'c1_0_mean': c1_0_mean,
                        'c1_1_mean': c1_1_mean,
                        'c1_2_mean': c1_2_mean,
                        'c2_0_mean': c2_0_mean,
                        'c2_1_mean': c2_1_mean,
                        'c2_2_mean': c2_2_mean}

        return calib_params


    def string_to_index(self,state):
        index = 0
        for digit in state:
            index = index * 3 + int(digit)
        return index

    def get_confusion(self):

        IQ_00 = self.POVM_dict['00']
        IQ_01 = self.POVM_dict['01']
        IQ_02 = self.POVM_dict['02']
        IQ_10 = self.POVM_dict['10']
        IQ_11 = self.POVM_dict['11']
        IQ_12 = self.POVM_dict['12']
        IQ_20 = self.POVM_dict['20']
        IQ_21 = self.POVM_dict['21']
        IQ_22 = self.POVM_dict['22']


        num_shots = len(IQ_00['IQ1'])
        confusion = np.zeros([9, 9])

        IQ_dict = {
            'IQ_00': IQ_00,
            'IQ_01': IQ_01,
            'IQ_02': IQ_02,
            'IQ_10': IQ_10,
            'IQ_11': IQ_11,
            'IQ_12': IQ_12,
            'IQ_20': IQ_20,
            'IQ_21': IQ_21,
            'IQ_22': IQ_22
        }

        for i in range(num_shots):

            for k,key in enumerate(IQ_dict.keys()):
                IQ_point = {'IQ1': IQ_dict[key]['IQ1'][i],
                            'IQ2': IQ_dict[key]['IQ2'][i],
                            }
                state_string = self.get_state(IQ_point)
                state_index = self.string_to_index(state_string)
                confusion[state_index][k] += 1 / num_shots

        return confusion

    def get_readout_fidelity(self):
        return np.trace(self.confusion)/9

    def get_state(self, IQ_point):

        c1_0_mean = self.calib_params['c1_0_mean']
        c1_1_mean = self.calib_params['c1_1_mean']
        c1_2_mean = self.calib_params['c1_2_mean']
        c2_0_mean = self.calib_params['c2_0_mean']
        c2_1_mean = self.calib_params['c2_1_mean']
        c2_2_mean = self.calib_params['c2_2_mean']

        s_q1 = find_state([np.real(IQ_point['IQ1']), np.imag(IQ_point['IQ1'])], c1_0_mean, c1_1_mean, c1_2_mean)
        s_q2 = find_state([np.real(IQ_point['IQ2']), np.imag(IQ_point['IQ2'])], c2_0_mean, c2_1_mean, c2_2_mean)

        return f"{s_q1}{s_q2}"


class DiscrimC2Q3L:

    def __init__(self, POVM_dict,POVM):

        self.POVM_dict = POVM_dict
        self.POVM = POVM
        self.calib_params = self.make_calib_params()
        self.confusion = self.get_confusion()
        self.readout_fidelity = self.get_readout_fidelity()

    def make_calib_params(self):

        IQ_000 = self.POVM_dict['000']
        IQ_010 = self.POVM_dict['010']
        IQ_020 = self.POVM_dict['020']
        IQ_100 = self.POVM_dict['100']
        IQ_110 = self.POVM_dict['110']
        IQ_120 = self.POVM_dict['120']
        IQ_200 = self.POVM_dict['200']
        IQ_210 = self.POVM_dict['210']
        IQ_220 = self.POVM_dict['220']

        IQ_001 = self.POVM_dict['001']
        IQ_011 = self.POVM_dict['011']
        IQ_021 = self.POVM_dict['021']
        IQ_101 = self.POVM_dict['101']
        IQ_111 = self.POVM_dict['111']
        IQ_121 = self.POVM_dict['121']
        IQ_201 = self.POVM_dict['201']
        IQ_211 = self.POVM_dict['211']
        IQ_221 = self.POVM_dict['221']

        #Resonator 1 labels
        I1_000 = np.real(IQ_000['IQ1'])
        Q1_000 = np.imag(IQ_000['IQ1'])
        I1_010 = np.real(IQ_010['IQ1'])
        Q1_010 = np.imag(IQ_010['IQ1'])
        I1_020 = np.real(IQ_020['IQ1'])
        Q1_020 = np.imag(IQ_020['IQ1'])
        I1_100 = np.real(IQ_100['IQ1'])
        Q1_100 = np.imag(IQ_100['IQ1'])
        I1_110 = np.real(IQ_110['IQ1'])
        Q1_110 = np.imag(IQ_110['IQ1'])
        I1_120 = np.real(IQ_120['IQ1'])
        Q1_120 = np.imag(IQ_120['IQ1'])
        I1_200 = np.real(IQ_200['IQ1'])
        Q1_200 = np.imag(IQ_200['IQ1'])
        I1_210 = np.real(IQ_210['IQ1'])
        Q1_210 = np.imag(IQ_210['IQ1'])
        I1_220 = np.real(IQ_220['IQ1'])
        Q1_220 = np.imag(IQ_220['IQ1'])

        I1_001 = np.real(IQ_001['IQ1'])
        Q1_001 = np.imag(IQ_001['IQ1'])
        I1_011 = np.real(IQ_011['IQ1'])
        Q1_011 = np.imag(IQ_011['IQ1'])
        I1_021 = np.real(IQ_021['IQ1'])
        Q1_021 = np.imag(IQ_021['IQ1'])
        I1_101 = np.real(IQ_101['IQ1'])
        Q1_101 = np.imag(IQ_101['IQ1'])
        I1_111 = np.real(IQ_111['IQ1'])
        Q1_111 = np.imag(IQ_111['IQ1'])
        I1_121 = np.real(IQ_121['IQ1'])
        Q1_121 = np.imag(IQ_121['IQ1'])
        I1_201 = np.real(IQ_201['IQ1'])
        Q1_201 = np.imag(IQ_201['IQ1'])
        I1_211 = np.real(IQ_211['IQ1'])
        Q1_211 = np.imag(IQ_211['IQ1'])
        I1_221 = np.real(IQ_221['IQ1'])
        Q1_221 = np.imag(IQ_221['IQ1'])

        # Resonator 2 labels
        I2_000 = np.real(IQ_000['IQ2'])
        Q2_000 = np.imag(IQ_000['IQ2'])
        I2_010 = np.real(IQ_010['IQ2'])
        Q2_010 = np.imag(IQ_010['IQ2'])
        I2_020 = np.real(IQ_020['IQ2'])
        Q2_020 = np.imag(IQ_020['IQ2'])
        I2_100 = np.real(IQ_100['IQ2'])
        Q2_100 = np.imag(IQ_100['IQ2'])
        I2_110 = np.real(IQ_110['IQ2'])
        Q2_110 = np.imag(IQ_110['IQ2'])
        I2_120 = np.real(IQ_120['IQ2'])
        Q2_120 = np.imag(IQ_120['IQ2'])
        I2_200 = np.real(IQ_200['IQ2'])
        Q2_200 = np.imag(IQ_200['IQ2'])
        I2_210 = np.real(IQ_210['IQ2'])
        Q2_210 = np.imag(IQ_210['IQ2'])
        I2_220 = np.real(IQ_220['IQ2'])
        Q2_220 = np.imag(IQ_220['IQ2'])

        I2_001 = np.real(IQ_001['IQ2'])
        Q2_001 = np.imag(IQ_001['IQ2'])
        I2_011 = np.real(IQ_011['IQ2'])
        Q2_011 = np.imag(IQ_011['IQ2'])
        I2_021 = np.real(IQ_021['IQ2'])
        Q2_021 = np.imag(IQ_021['IQ2'])
        I2_101 = np.real(IQ_101['IQ2'])
        Q2_101 = np.imag(IQ_101['IQ2'])
        I2_111 = np.real(IQ_111['IQ2'])
        Q2_111 = np.imag(IQ_111['IQ2'])
        I2_121 = np.real(IQ_121['IQ2'])
        Q2_121 = np.imag(IQ_121['IQ2'])
        I2_201 = np.real(IQ_201['IQ2'])
        Q2_201 = np.imag(IQ_201['IQ2'])
        I2_211 = np.real(IQ_211['IQ2'])
        Q2_211 = np.imag(IQ_211['IQ2'])
        I2_221 = np.real(IQ_221['IQ2'])
        Q2_221 = np.imag(IQ_221['IQ2'])

        # Resonator c labels
        Ic_000 = np.real(IQ_000['IQc'])
        Qc_000 = np.imag(IQ_000['IQc'])
        Ic_010 = np.real(IQ_010['IQc'])
        Qc_010 = np.imag(IQ_010['IQc'])
        Ic_020 = np.real(IQ_020['IQc'])
        Qc_020 = np.imag(IQ_020['IQc'])
        Ic_100 = np.real(IQ_100['IQc'])
        Qc_100 = np.imag(IQ_100['IQc'])
        Ic_110 = np.real(IQ_110['IQc'])
        Qc_110 = np.imag(IQ_110['IQc'])
        Ic_120 = np.real(IQ_120['IQc'])
        Qc_120 = np.imag(IQ_120['IQc'])
        Ic_200 = np.real(IQ_200['IQc'])
        Qc_200 = np.imag(IQ_200['IQc'])
        Ic_210 = np.real(IQ_210['IQc'])
        Qc_210 = np.imag(IQ_210['IQc'])
        Ic_220 = np.real(IQ_220['IQc'])
        Qc_220 = np.imag(IQ_220['IQc'])

        Ic_001 = np.real(IQ_001['IQc'])
        Qc_001 = np.imag(IQ_001['IQc'])
        Ic_011 = np.real(IQ_011['IQc'])
        Qc_011 = np.imag(IQ_011['IQc'])
        Ic_021 = np.real(IQ_021['IQc'])
        Qc_021 = np.imag(IQ_021['IQc'])
        Ic_101 = np.real(IQ_101['IQc'])
        Qc_101 = np.imag(IQ_101['IQc'])
        Ic_111 = np.real(IQ_111['IQc'])
        Qc_111 = np.imag(IQ_111['IQc'])
        Ic_121 = np.real(IQ_121['IQc'])
        Qc_121 = np.imag(IQ_121['IQc'])
        Ic_201 = np.real(IQ_201['IQc'])
        Qc_201 = np.imag(IQ_201['IQc'])
        Ic_211 = np.real(IQ_211['IQc'])
        Qc_211 = np.imag(IQ_211['IQc'])
        Ic_221 = np.real(IQ_221['IQc'])
        Qc_221 = np.imag(IQ_221['IQc'])

        c1_0_mean = [np.mean([I1_000, I1_010, I1_020,I1_001, I1_011, I1_021]), np.mean([Q1_000, Q1_010, Q1_020,Q1_001, Q1_011, Q1_021])]
        c1_1_mean = [np.mean([I1_100, I1_110, I1_120,I1_101, I1_111, I1_121]), np.mean([Q1_100, Q1_110, Q1_120,Q1_101, Q1_111, Q1_121])]
        c1_2_mean = [np.mean([I1_200, I1_210, I1_220,I1_201, I1_211, I1_221]), np.mean([Q1_200, Q1_210, Q1_220,Q1_201, Q1_211, Q1_221])]
        c2_0_mean = [np.mean([I2_000, I2_100, I2_200,I2_001, I2_101, I2_201]), np.mean([Q2_000, Q2_100, Q2_200,Q2_001, Q2_101, Q2_201])]
        c2_1_mean = [np.mean([I2_010, I2_110, I2_210,I2_011, I2_111, I2_211]), np.mean([Q2_010, Q2_110, Q2_210,Q2_011, Q2_111, Q2_211])]
        c2_2_mean = [np.mean([I2_020, I2_120, I2_220,I2_021, I2_121, I2_221]), np.mean([Q2_020, Q2_120, Q2_220,Q2_021, Q2_121, Q2_221])]
        cc_0_mean = [np.mean([Ic_000, Ic_010, Ic_020, Ic_100, Ic_110, Ic_120, Ic_200, Ic_210, Ic_220]), np.mean([Qc_000, Qc_010, Qc_020, Qc_100, Qc_110, Qc_120, Qc_200, Qc_210, Qc_220])]
        cc_1_mean = [np.mean([Ic_001, Ic_011, Ic_021, Ic_101, Ic_111, Ic_121, Ic_201, Ic_211, Ic_221]), np.mean([Qc_001, Qc_011, Qc_021, Qc_101, Qc_111, Qc_121, Qc_201, Qc_211, Qc_221])]


        calib_params = {'c1_0_mean': c1_0_mean,
                        'c1_1_mean': c1_1_mean,
                        'c1_2_mean': c1_2_mean,
                        'c2_0_mean': c2_0_mean,
                        'c2_1_mean': c2_1_mean,
                        'c2_2_mean': c2_2_mean,
                        'cc_0_mean': cc_0_mean,
                        'cc_1_mean': cc_1_mean}

        return calib_params

    def string_to_index(self,state): #takes a state as a string
        index = 0
        for digit in state[:-1]:
            index = index * 3 + int(digit)
        index = index + 9 * int(state[-1])
        return index

    def get_confusion(self):

        IQ_000 = self.POVM_dict['000']
        IQ_010 = self.POVM_dict['010']
        IQ_020 = self.POVM_dict['020']
        IQ_100 = self.POVM_dict['100']
        IQ_110 = self.POVM_dict['110']
        IQ_120 = self.POVM_dict['120']
        IQ_200 = self.POVM_dict['200']
        IQ_210 = self.POVM_dict['210']
        IQ_220 = self.POVM_dict['220']

        IQ_001 = self.POVM_dict['001']
        IQ_011 = self.POVM_dict['011']
        IQ_021 = self.POVM_dict['021']
        IQ_101 = self.POVM_dict['101']
        IQ_111 = self.POVM_dict['111']
        IQ_121 = self.POVM_dict['121']
        IQ_201 = self.POVM_dict['201']
        IQ_211 = self.POVM_dict['211']
        IQ_221 = self.POVM_dict['221']




        num_shots = len(IQ_000['IQ1'])
        confusion = np.zeros([18, 18])

        IQ_dict = {
            'IQ_000': IQ_000,
            'IQ_010': IQ_010,
            'IQ_020': IQ_020,
            'IQ_100': IQ_100,
            'IQ_110': IQ_110,
            'IQ_120': IQ_120,
            'IQ_200': IQ_200,
            'IQ_210': IQ_210,
            'IQ_220': IQ_220,
            'IQ_001': IQ_001,
            'IQ_011': IQ_011,
            'IQ_021': IQ_021,
            'IQ_101': IQ_101,
            'IQ_111': IQ_111,
            'IQ_121': IQ_121,
            'IQ_201': IQ_201,
            'IQ_211': IQ_211,
            'IQ_221': IQ_221
        }

        for i in range(num_shots):

            for k,key in enumerate(IQ_dict.keys()):
                IQ_point = {'IQ1': IQ_dict[key]['IQ1'][i],
                            'IQ2': IQ_dict[key]['IQ2'][i],
                            'IQc': IQ_dict[key]['IQc'][i]
                            }

                state_string = self.get_state(IQ_point)
                state_index = self.string_to_index(state_string)
                confusion[state_index][k] += 1 / num_shots


        return confusion

    def get_readout_fidelity(self):

        return np.trace(self.confusion)/18

    def get_state(self, IQ_point):

        c1_0_mean = self.calib_params['c1_0_mean']
        c1_1_mean = self.calib_params['c1_1_mean']
        c1_2_mean = self.calib_params['c1_2_mean']
        c2_0_mean = self.calib_params['c2_0_mean']
        c2_1_mean = self.calib_params['c2_1_mean']
        c2_2_mean = self.calib_params['c2_2_mean']
        cc_0_mean = self.calib_params['cc_0_mean']
        cc_1_mean = self.calib_params['cc_1_mean']

        s_q1 = find_state([np.real(IQ_point['IQ1']),np.imag(IQ_point['IQ1'])], c1_0_mean, c1_1_mean, c1_2_mean)
        s_q2 = find_state([np.real(IQ_point['IQ2']),np.imag(IQ_point['IQ2'])], c2_0_mean, c2_1_mean, c2_2_mean)
        s_qc = find_state([np.real(IQ_point['IQc']),np.imag(IQ_point['IQc'])], cc_0_mean, cc_1_mean)

        return f"{s_q1}{s_q2}{s_qc}"


class Discrim_G_C2Q3L:

    def __init__(self, POVM_dict, POVM):

        self.POVM_dict = POVM_dict #it has all the states, POVM labels are stored in POVM
        self.POVM = POVM
        self.calib_params = self.make_calib_params()
        self.confusion = self.get_confusion()
        self.readout_fidelity = self.get_readout_fidelity()

    def make_calib_params(self):

        IQ_abc = {}
        c1_0_I,c1_0_Q,c1_1_I,c1_1_Q,c1_2_I,c1_2_Q,c2_0_I,c2_0_Q,c2_1_I,c2_1_Q,c2_2_I,c2_2_Q,cc_0_I,cc_0_Q,cc_1_I,cc_1_Q = ([] for i in range(16))

        for key in self.POVM_dict.keys():
            IQ_data = self.POVM_dict[key]
            # IQ_abc['IQ_' + key] = IQ_data
            IQ_abc['IQ_' + key] = {'I1_' + key: np.real(IQ_data['IQ1']),
                                   'Q1_' + key: np.imag(IQ_data['IQ1']),
                                   'I2_' + key: np.real(IQ_data['IQ2']),
                                   'Q2_' + key: np.imag(IQ_data['IQ2']),
                                   'Ic_' + key: np.real(IQ_data['IQc']),
                                   'Qc_' + key: np.imag(IQ_data['IQc']),
                                   }
            if key[0] == '0':
                c1_0_I.append(np.real(IQ_data['IQ1']))
                c1_0_Q.append(np.imag(IQ_data['IQ1']))
            if key[0] == '1':
                c1_1_I.append(np.real(IQ_data['IQ1']))
                c1_1_Q.append(np.imag(IQ_data['IQ1']))
            if key[0] == '2':
                c1_2_I.append(np.real(IQ_data['IQ1']))
                c1_2_Q.append(np.imag(IQ_data['IQ1']))

            if key[1] == '0':
                c2_0_I.append(np.real(IQ_data['IQ2']))
                c2_0_Q.append(np.imag(IQ_data['IQ2']))
            if key[1] == '1':
                c2_1_I.append(np.real(IQ_data['IQ2']))
                c2_1_Q.append(np.imag(IQ_data['IQ2']))
            if key[1] == '2':
                c2_2_I.append(np.real(IQ_data['IQ2']))
                c2_2_Q.append(np.imag(IQ_data['IQ2']))

            if key[2] == '0':
                cc_0_I.append(np.real(IQ_data['IQc']))
                cc_0_Q.append(np.imag(IQ_data['IQc']))
            if key[2] == '1':
                cc_1_I.append(np.real(IQ_data['IQc']))
                cc_1_Q.append(np.imag(IQ_data['IQc']))

        c1_0_mean = [np.mean(c1_0_I), np.mean(c1_0_Q)]
        c1_1_mean = [np.mean(c1_1_I), np.mean(c1_1_Q)]
        c1_2_mean = [np.mean(c1_2_I), np.mean(c1_2_Q)]
        c2_0_mean = [np.mean(c2_0_I), np.mean(c2_0_Q)]
        c2_1_mean = [np.mean(c2_1_I), np.mean(c2_1_Q)]
        c2_2_mean = [np.mean(c2_2_I), np.mean(c2_2_Q)]
        cc_0_mean = [np.mean(cc_0_I), np.mean(cc_0_Q)]
        cc_1_mean = [np.mean(cc_1_I), np.mean(cc_1_Q)]

        calib_params = {'c1_0_mean': c1_0_mean,
                        'c1_1_mean': c1_1_mean,
                        'c1_2_mean': c1_2_mean,
                        'c2_0_mean': c2_0_mean,
                        'c2_1_mean': c2_1_mean,
                        'c2_2_mean': c2_2_mean,
                        'cc_0_mean': cc_0_mean,
                        'cc_1_mean': cc_1_mean}
        # print(calib_params)


        return calib_params

    def string_to_index(self,state): #takes a state as a string
        if state == None:
            raise ValueError("The State found is not in POVMs")

        for i,p in enumerate(self.POVM.keys()):
            if state == p:
                index = i
        return index

    def get_confusion(self):
        IQ_dict = {} #it's an old-way POVM Dict
        for p in self.POVM.keys():
            IQ1 = np.zeros(0)
            IQ2 = np.zeros(0)
            IQc = np.zeros(0)

            for s in self.POVM[p]:
                IQ1 = concatenate(IQ1, self.POVM_dict[s]['IQ1'])
                IQ2 = concatenate(IQ2, self.POVM_dict[s]['IQ2'])
                IQc = concatenate(IQc, self.POVM_dict[s]['IQc'])

            IQ_dict[p] =   {'IQ1': IQ1,
                            'IQ2': IQ2,
                            'IQc': IQc}

        num_shots = np.zeros(len(IQ_dict))
        for i,p in enumerate(IQ_dict.keys()):
            num_shots[i] = len(IQ_dict[p]['IQ1'])

        # num_shots = len(self.POVM_dict['000']['IQ1'])
        confusion = np.zeros([len(self.POVM), len(self.POVM)])

        for k,key in enumerate(IQ_dict.keys()):
            for i in range(int(num_shots[k])):
                IQ_point = {'IQ1': IQ_dict[key]['IQ1'][i],
                            'IQ2': IQ_dict[key]['IQ2'][i],
                            'IQc': IQ_dict[key]['IQc'][i]
                            }

                state_string = self.get_state(IQ_point)
                state_index = self.string_to_index(state_string)
                confusion[state_index][k] += 1 / num_shots[k]


        return confusion

    def get_readout_fidelity(self):

        return np.trace(self.confusion)/len(self.POVM)

    def get_state(self, IQ_point):

        c1_0_mean = self.calib_params['c1_0_mean']
        c1_1_mean = self.calib_params['c1_1_mean']
        c1_2_mean = self.calib_params['c1_2_mean']
        c2_0_mean = self.calib_params['c2_0_mean']
        c2_1_mean = self.calib_params['c2_1_mean']
        c2_2_mean = self.calib_params['c2_2_mean']
        cc_0_mean = self.calib_params['cc_0_mean']
        cc_1_mean = self.calib_params['cc_1_mean']

        s_q1 = find_state([np.real(IQ_point['IQ1']),np.imag(IQ_point['IQ1'])], c1_0_mean, c1_1_mean, c1_2_mean)
        s_q2 = find_state([np.real(IQ_point['IQ2']),np.imag(IQ_point['IQ2'])], c2_0_mean, c2_1_mean, c2_2_mean)
        s_qc = find_state([np.real(IQ_point['IQc']),np.imag(IQ_point['IQc'])], cc_0_mean, cc_1_mean)

        state = f"{s_q1}{s_q2}{s_qc}"
        # print('the means are cavity1(2L), cavity2(2L), cavityc(2L)',c1_0_mean, c1_1_mean,c2_0_mean, c2_1_mean,cc_0_mean, cc_1_mean )
        # print(state)

        for p in self.POVM.keys():
            for s in self.POVM[p]:
                if s == state:
                    ver = p
                    return ver

        # for p in self.POVM.keys(): # for only two states
        #     for s in self.POVM[p]:
        #         if s[:-1] == state[:-1]:
        #             ver = p
        #             return ver


class DiscrimCL:

    def __init__(self, POVM_dict):

        self.POVM_dict = POVM_dict
        self.calib_params = self.make_calib_params()
        self.confusion = self.get_confusion()
        self.readout_fidelity = self.get_readout_fidelity()

    def make_calib_params(self):

        IQ_C = self.POVM_dict['C']
        IQ_L = self.POVM_dict['L']


        #Resonator 1 labels
        I1_C = np.real(IQ_C['IQ1'])
        Q1_C = np.imag(IQ_C['IQ1'])
        I1_L = np.real(IQ_L['IQ1'])
        Q1_L = np.imag(IQ_L['IQ1'])

        # Resonator 1 labels
        I2_C = np.real(IQ_C['IQ2'])
        Q2_C = np.imag(IQ_C['IQ2'])
        I2_L = np.real(IQ_L['IQ2'])
        Q2_L = np.imag(IQ_L['IQ2'])

        # Resonator 1 labels
        Ic_C = np.real(IQ_C['IQc'])
        Qc_C = np.imag(IQ_C['IQc'])
        Ic_L = np.real(IQ_L['IQc'])
        Qc_L = np.imag(IQ_L['IQc'])



        c1_C_mean = [np.mean([I1_C]), np.mean([Q1_C])]
        c1_L_mean = [np.mean([I1_L]), np.mean([Q1_L])]
        c2_C_mean = [np.mean([I2_C]), np.mean([Q2_C])]
        c2_L_mean = [np.mean([I2_L]), np.mean([Q2_L])]
        cc_C_mean = [np.mean([Ic_C]), np.mean([Qc_C])]
        cc_L_mean = [np.mean([Ic_L]), np.mean([Qc_L])]


        calib_params = {'c1_C_mean': c1_C_mean,
                        'c1_L_mean': c1_L_mean,
                        'c2_C_mean': c2_C_mean,
                        'c2_L_mean': c2_L_mean,
                        'cc_C_mean': cc_C_mean,
                        'cc_L_mean': cc_L_mean}

        return calib_params

    def string_to_index(self,state): #takes a state as a string
        if state == 'C':
            index = 0
        else:
            index = 1
        return index

    def get_confusion(self):

        IQ_C = self.POVM_dict['C']
        IQ_L = self.POVM_dict['L']



        num_shots = len(IQ_C['IQ1'])
        confusion = np.zeros([2, 2])

        IQ_dict = {
            'IQ_C': IQ_C,
            'IQ_L': IQ_L
        }



        for i in range(num_shots):

            for k,key in enumerate(IQ_dict.keys()):
                IQ_point = {'IQ1': IQ_dict[key]['IQ1'][i],
                            'IQ2': IQ_dict[key]['IQ2'][i],
                            'IQc': IQ_dict[key]['IQc'][i]
                            }

                state_string = self.get_state(IQ_point)
                state_index = self.string_to_index(state_string)
                confusion[state_index][k] += 1 / num_shots

        return confusion

    def get_readout_fidelity(self):

        return np.trace(self.confusion)/2

    def get_state(self, IQ_point):

        c1_C_mean = self.calib_params['c1_C_mean']
        c1_L_mean = self.calib_params['c1_L_mean']
        c2_C_mean = self.calib_params['c2_C_mean']
        c2_L_mean = self.calib_params['c2_L_mean']
        cc_C_mean = self.calib_params['cc_C_mean']
        cc_L_mean = self.calib_params['cc_L_mean']

        s_q1 = find_state([np.real(IQ_point['IQ1']),np.imag(IQ_point['IQ1'])], c1_C_mean, c1_L_mean)
        s_q2 = find_state([np.real(IQ_point['IQ2']),np.imag(IQ_point['IQ2'])], c2_C_mean, c2_L_mean)
        s_qc = find_state([np.real(IQ_point['IQc']),np.imag(IQ_point['IQc'])], cc_C_mean, cc_L_mean)

        state = f"{s_q1}{s_q2}{s_qc}"
        if state == '000':
            ver = "C"
        else:
            ver = "L"

        # if state == '000' or state == '010' or state == '100' or state == '110':
        #     ver = "C"
        # else:
        #     ver = "L"


        return ver


class DiscrimCLX:

    def __init__(self, POVM_dict):

        self.POVM_dict = POVM_dict
        self.calib_params = self.make_calib_params()
        self.confusion = self.get_confusion()
        self.readout_fidelity = self.get_readout_fidelity()

    def make_calib_params(self):

        IQ_C    = self.POVM_dict['C']
        IQ_X2nn = self.POVM_dict['X2nn']
        IQ_Xn2n = self.POVM_dict['Xn2n']
        IQ_Xnn1 = self.POVM_dict['Xnn1']
        IQ_X22n = self.POVM_dict['X22n']
        IQ_X2n1 = self.POVM_dict['X2n1']
        IQ_Xn21 = self.POVM_dict['Xn21']
        IQ_X221 = self.POVM_dict['X221']




        #Resonator 1 labels
        I1_C = np.real(IQ_C['IQ1'])
        Q1_C = np.imag(IQ_C['IQ1'])
        I1_X2nn = np.real(IQ_X2nn['IQ1'])
        Q1_X2nn = np.imag(IQ_X2nn['IQ1'])
        I1_Xn2n = np.real(IQ_Xn2n['IQ1'])
        Q1_Xn2n = np.imag(IQ_Xn2n['IQ1'])
        I1_Xnn1 = np.real(IQ_Xnn1['IQ1'])
        Q1_Xnn1 = np.imag(IQ_Xnn1['IQ1'])
        I1_X22n = np.real(IQ_X22n['IQ1'])
        Q1_X22n = np.imag(IQ_X22n['IQ1'])
        I1_X2n1 = np.real(IQ_X2n1['IQ1'])
        Q1_X2n1 = np.imag(IQ_X2n1['IQ1'])
        I1_Xn21 = np.real(IQ_Xn21['IQ1'])
        Q1_Xn21 = np.imag(IQ_Xn21['IQ1'])
        I1_X221 = np.real(IQ_X221['IQ1'])
        Q1_X221 = np.imag(IQ_X221['IQ1'])
        
        
        

        # Resonator 1 labels
        I2_C = np.real(IQ_C['IQ2'])
        Q2_C = np.imag(IQ_C['IQ2'])
        I2_X2nn = np.real(IQ_X2nn['IQ2'])
        Q2_X2nn = np.imag(IQ_X2nn['IQ2'])
        I2_Xn2n = np.real(IQ_Xn2n['IQ2'])
        Q2_Xn2n = np.imag(IQ_Xn2n['IQ2'])
        I2_Xnn1 = np.real(IQ_Xnn1['IQ2'])
        Q2_Xnn1 = np.imag(IQ_Xnn1['IQ2'])
        I2_X22n = np.real(IQ_X22n['IQ2'])
        Q2_X22n = np.imag(IQ_X22n['IQ2'])
        I2_X2n1 = np.real(IQ_X2n1['IQ2'])
        Q2_X2n1 = np.imag(IQ_X2n1['IQ2'])
        I2_Xn21 = np.real(IQ_Xn21['IQ2'])
        Q2_Xn21 = np.imag(IQ_Xn21['IQ2'])
        I2_X221 = np.real(IQ_X221['IQ2'])
        Q2_X221 = np.imag(IQ_X221['IQ2'])

        # Resonator 1 labels
        Ic_C = np.real(IQ_C['IQc'])
        Qc_C = np.imag(IQ_C['IQc'])
        Ic_X2nn = np.real(IQ_X2nn['IQc'])
        Qc_X2nn = np.imag(IQ_X2nn['IQc'])
        Ic_Xn2n = np.real(IQ_Xn2n['IQc'])
        Qc_Xn2n = np.imag(IQ_Xn2n['IQc'])
        Ic_Xnn1 = np.real(IQ_Xnn1['IQc'])
        Qc_Xnn1 = np.imag(IQ_Xnn1['IQc'])
        Ic_X22n = np.real(IQ_X22n['IQc'])
        Qc_X22n = np.imag(IQ_X22n['IQc'])
        Ic_X2n1 = np.real(IQ_X2n1['IQc'])
        Qc_X2n1 = np.imag(IQ_X2n1['IQc'])
        Ic_Xn21 = np.real(IQ_Xn21['IQc'])
        Qc_Xn21 = np.imag(IQ_Xn21['IQc'])
        Ic_X221 = np.real(IQ_X221['IQc'])
        Qc_X221 = np.imag(IQ_X221['IQc'])



        c1_C_mean = [np.mean([I1_C]), np.mean([Q1_C])]
        c1_X2nn_mean = [np.mean([I1_L]), np.mean([Q1_L])]
        c2_C_mean = [np.mean([I2_C]), np.mean([Q2_C])]
        c2_L_mean = [np.mean([I2_L]), np.mean([Q2_L])]
        cc_C_mean = [np.mean([Ic_C]), np.mean([Qc_C])]
        cc_L_mean = [np.mean([Ic_L]), np.mean([Qc_L])]


        calib_params = {'c1_C_mean': c1_C_mean,
                        'c1_L_mean': c1_L_mean,
                        'c2_C_mean': c2_C_mean,
                        'c2_L_mean': c2_L_mean,
                        'cc_C_mean': cc_C_mean,
                        'cc_L_mean': cc_L_mean}

        return calib_params

    def string_to_index(self,state): #takes a state as a string
        if state == 'C':
            index = 0
        else:
            index = 1
        return index

    def get_confusion(self):

        IQ_C = self.POVM_dict['C']
        IQ_L = self.POVM_dict['L']



        num_shots = len(IQ_C['IQ1'])
        confusion = np.zeros([2, 2])

        IQ_dict = {
            'IQ_C': IQ_C,
            'IQ_L': IQ_L
        }

        for i in range(num_shots):

            for k,key in enumerate(IQ_dict.keys()):
                IQ_point = {'IQ1': IQ_dict[key]['IQ1'][i],
                            'IQ2': IQ_dict[key]['IQ2'][i],
                            'IQc': IQ_dict[key]['IQc'][i]
                            }

                state_string = self.get_state(IQ_point)
                state_index = self.string_to_index(state_string)
                confusion[state_index][k] += 1 / num_shots

        return confusion

    def get_readout_fidelity(self):

        return np.trace(self.confusion)/2

    def get_state(self, IQ_point):

        c1_C_mean = self.calib_params['c1_C_mean']
        c1_L_mean = self.calib_params['c1_L_mean']
        c2_C_mean = self.calib_params['c2_C_mean']
        c2_L_mean = self.calib_params['c2_L_mean']
        cc_C_mean = self.calib_params['cc_C_mean']
        cc_L_mean = self.calib_params['cc_L_mean']

        s_q1 = find_state([np.real(IQ_point['IQ1']),np.imag(IQ_point['IQ1'])], c1_C_mean, c1_L_mean)
        s_q2 = find_state([np.real(IQ_point['IQ2']),np.imag(IQ_point['IQ2'])], c2_C_mean, c2_L_mean)
        s_qc = find_state([np.real(IQ_point['IQc']),np.imag(IQ_point['IQc'])], cc_C_mean, cc_L_mean)

        state = f"{s_q1}{s_q2}{s_qc}"
        if state == '000':
            ver = "C"
        else:
            ver = "L"


        return ver


def find_blob_mean(IQ_trace, axis=None):
    if axis:
        return np.mean(np.real(IQ_trace), axis) + 1j * np.mean(np.imag(IQ_trace), axis)
    else:
        return np.mean(np.real(IQ_trace)) + 1j * np.mean(np.imag(IQ_trace))

def find_blob_median(IQ_trace, axis=None):
    if axis:
        return np.median(np.real(IQ_trace), axis) + 1j * np.median(np.imag(IQ_trace), axis)
    else:
        return np.median(np.real(IQ_trace)) + 1j * np.median(np.imag(IQ_trace))


def concatenate(IQ_trace_1, IQ_trace_2):
    return np.concatenate((IQ_trace_1, IQ_trace_2))


def find_threshold(IQ_trace_g, IQ_trace_e):
    ground_projections = np.sort(np.real(IQ_trace_g))
    excited_projections = np.sort(np.real(IQ_trace_e))

    min_threshold = np.min(concatenate(ground_projections, excited_projections))
    max_threshold = np.max(concatenate(ground_projections, excited_projections))

    ground_mean = np.mean(ground_projections)
    excited_mean = np.mean(excited_projections)

    if ground_mean < excited_mean:
        left_blob = ground_projections
        right_blob = excited_projections
    else:
        left_blob = excited_projections
        right_blob = ground_projections

    num_thresholds = 100
    num_items = len(right_blob)

    F = np.zeros(num_thresholds)
    thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)

    for i, t in enumerate(thresholds):

        F_left = len(right_blob)
        F_right = 0

        for b in left_blob:

            if t > b:
                F_left = F_left - 1
            else:
                break

        for b in right_blob:

            if t > b:
                F_right = F_right + 1
            else:
                break

        F[i] = 1 - F_left / (2*num_items) - F_right / (2*num_items)

    return thresholds, F


def find_state(IQ, *q_IQ):

    # IQ is the state you are checking
    # q_IQ is a list of blob means

    distances = [(IQ[0]-iq[0]) ** 2 + (IQ[1]-iq[1]) ** 2 for iq in q_IQ]
    state = np.nanargmin(distances) #ignored stupid NaN values
    return int(state)


def find_state_1q_nL(I, Q, calib_params):
    """
    Generalized to N-level single qubit.

    I, Q: arrays of identical arbitrary shape
    calib_params: list/tuple of N IQ centroids, each like [I_mean, Q_mean]

    Returns:
        ss_array with same shape as I and Q
    """
    I = np.asarray(I)
    Q = np.asarray(Q)

    if I.shape != Q.shape:
        raise ValueError("I and Q must have the same shape")

    flat_I = I.ravel()
    flat_Q = Q.ravel()

    ss_flat = np.empty(flat_I.shape, dtype=int)

    for idx, (i, q) in enumerate(zip(flat_I, flat_Q)):
        ss_flat[idx] = find_state([i, q], *calib_params)

    return ss_flat.reshape(I.shape)

def discrim_2Q_2L(calib_params,shape,I_1,Q_1,I_2,Q_2,I_c,Q_c):

    ss_thresh_1 = calib_params[0]
    ss_factor_1 = calib_params[1]
    ss_theta_1 = calib_params[2]
    ss_thresh_2 = calib_params[3]
    ss_factor_2 = calib_params[4]
    ss_theta_2 = calib_params[5]

    IQ_1 = ss_factor_1 * np.real(np.exp(-1j * ss_theta_1) * (I_1 + 1j * Q_1))
    IQ_2 = ss_factor_2 * np.real(np.exp(-1j * ss_theta_2) * (I_2 + 1j * Q_2))
    IQ_10_1 = np.zeros(len(IQ_1))
    IQ_10_2 = np.zeros(len(IQ_2))

    for i, v in enumerate(IQ_1):
        if v > ss_thresh_1:
            IQ_10_1[i] = 1

    for i, v in enumerate(IQ_2):
        if v > ss_thresh_2:
            IQ_10_2[i] = 1

    IQ_10 = np.zeros(len(IQ_1),dtype=int)

    for i in range(len(IQ_1)):
        IQ_10[i] = IQ_10_1[i]+2*IQ_10_2[i]

    ss_array = IQ_10.reshape(shape)

    num_shots = np.shape(ss_array)[0]
    num_circuits = np.shape(ss_array)[1]

    counts = np.zeros([4,num_circuits],dtype=int)

    for i in range(num_shots):
        for j in range(num_circuits):
            counts[ss_array[i][j]][j] += 1

    return counts