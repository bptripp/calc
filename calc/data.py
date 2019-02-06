import os
# import sys
# sys.path.append('/Users/bptripp/lib/multi-area-model/')
import inspect
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Data:
    def __init__(self):
        iac = InterAreaConnections()
        self.FLNe = iac.get_interpolated_FLNe_Schmidt()
        self.SLN = iac.get_interpolated_SLN_Schmidt()
        self.connections = iac.get_connectivity_grid()
        self.sd = SchmidtData()

        histogram = np.array(FS09_synapses_per_connection)
        self.synapses_per_connection = np.dot(histogram[:,0], histogram[:,1]) / np.sum(histogram[:,1])

        with open('/Users/bptripp/lib/multi-area-model/multiarea_model/data_multiarea/default_Data_Model_.json', 'r') as f:
            self.dat = json.load(f)

    def get_areas(self):
        """
        :return: List of visual cortical areas in the FV91 parcellation.
        """
        return areas_FV91

    def get_layers(self, area):
        """
        :param area: A visual area
        :return: List of layers in that area, e.g. '1', '2/3', etc.; these are fairly
             but not completely consistent across visual areas
        """
        return get_layers(area)

    def get_num_neurons(self, area, layer):
        """
        :param area: A visual area
        :param layer: Cortical layer
        :return: Estimated number of excitatory neurons in the given area/layer per hemisphere; we
            assume convolutional units are similar to excitatory neurons based on Parisien et al. (2008).
        """
        return self.sd.neuron_numbers(area, layer)

    def get_receptive_field_size(self, area):
        """
        :param area: A visual area
        :return: RF width at 5 degrees eccentricity, if available, otherwise None
        """
        return get_RF_size(area)

    def get_inputs_per_neuron(self, area, source_layer, target_layer):
        """
        :param area: A visual area
        :param source_layer: A cortical layer; source of an inter-laminar connection
        :param target_layer: Another cortical layer; target of same inter-laminar connection
        :return: Number of inputs in this connection per post-synaptic neuron
        """
        spn = sd.interlaminar_synapses_per_neuron(area, source_layer, target_layer)
        return spn / self.synapses_per_connection

    def get_extrinsic_inputs(self, area, target_layer):
        """
        :param area: A visual area
        :param target_layer: A cortical layer
        :return: Estimated number of feedforward inputs per neuron from outside this area
        """
        if area == 'V1':
            return 8 # estimate from Garcia-Marin et al. (2017)
        else:
            spn = sd.interarea_synapses_per_neuron(area, target_layer)
            return spn / self.synapses_per_connection

    def get_source_areas(self, target_area, feedforward_only=False):
        c = self.connections[areas_FV91.index(target_area)]

        result = []
        for i in range(len(areas_FV91)):
            if c[i]:
                source_area = areas_FV91[i]
                if not feedforward_only or FV91_hierarchy[target_area] > FV91_hierarchy[source_area]:
                    result.append(source_area)
        return result

    def get_FLNe(self, source_area, target_area):
        source_index = areas_FV91.index(source_area)
        target_index = areas_FV91.index(target_area)
        return self.FLNe[target_index, source_index]

    def get_SLN(self, source_area, target_area):
        source_index = areas_FV91.index(source_area)
        target_index = areas_FV91.index(target_area)
        return self.SLN[target_index, source_index]


areas_M132 = ['???', '1', '2', '3', '5', '9', '10', '11', '12', '13', '14', '23', '25', '31', '32', '44', '24a', '24b', '24c', '24d', '29/30', '45A', '45B', '46d', '46v', '7A', '7B', '7m', '7op', '8B', '8l', '8m', '8r', '9/46d', '9/46v', 'AIP', 'CORE', 'DP', 'ENTORHINAL', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'FST', 'Gu', 'INSULA', 'IPa', 'LB', 'LIP', 'MB', 'MIP', 'MST', 'MT', 'OPAI', 'OPRO', 'Parainsula', 'PBc', 'PBr', 'PERIRHINAL', 'PGa', 'PIP', 'PIRIFORM', 'ProM', 'Pro.St', 'SII', 'STPc', 'STPi', 'STPr', 'SUBICULUM', 'TEad', 'TEa/ma', 'TEa/mp', 'TEav', 'TEMPORAL_POLE', 'TEO', 'TEOm', 'TEpd', 'TEpv', 'TH/TF', 'TPt', 'V1', 'V2', 'V3', 'V3A', 'V4', 'V4t', 'V6', 'V6A', 'VIP']


areas_FV91 = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd', 'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd', 'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp', 'STPa', '46', 'AITd', 'TH']


class InterAreaConnections:
    """
    Provides data on connections between cortical areas, from retrograde tracer injections.
    This includes for each area that provides input to an injected area, the fraction of
    the total labelled neurons extrinsic to injection site (FLNe) and % of source neurons
    in supragranular layers (%SLN).

    The data sources are CoCoMac 2.0 and Markov et al. (2014). Markov et al. provide FLNe and
    %SLN for inputs to 29 injection sites using the M132 parcellation. Following Schmidt et al.
    (2018), we attempt to translate this dataset into the FV91 parcellation. We use FV91
    areas in which each injection was located (mapped by Schmidt et al.) Also following Schmidt,
    we convert sources from M132 to FV91 using overlap fractions of areas in each parcellation
    mapped to the F99 cortical surface. The overlap fractions are available from CoCoMac.

    We use trends in the data to fill in missing values for areas not injected by Markov et al.
    We only consider connections that are identified in CoCoMac 2.0.
    """
    def __init__(self, cocomac=None, markov=None):
        if not cocomac:
            cocomac = CoCoMac()
        if not markov:
            markov = Markov()

        self.cocomac = cocomac
        self.markov = markov
        self.areas = areas_FV91

        with open('./data_files/schmidt/default_Data_Model_.json', 'r') as f:
            self.dat = json.load(f)

    def get_connectivity_grid(self):
        """
        :return: A square matrix of connectivity between FV91 visual areas, with ones
            for connections that exist in CoCoMac 2.0 and zeros elsewhere. Target areas
            are in rows and sources are in columns.
        """
        n = len(self.areas)
        grid = np.zeros((n, n))
        for i in range(n):
            for source in self.cocomac.get_source_areas(self.areas[i]):
                if source in self.areas and source != self.areas[i]:
                    grid[i,self.areas.index(source)] = 1
        return grid

    def get_M132_FLNe(self, target):
        """
        :param target: A M132 area
        :return: a dictionary with M132 connected areas as keys and FLNe as values
        """
        result = dict()
        for source in self.markov.get_sources(target):
            result[source] = self.markov.get_FLNe(source, target)
        return result

    def get_interpolated_SLN_Schmidt(self):
        n = len(self.areas)
        result = np.nan * np.zeros((n,n))
        for i in range(n):
            d = self.dat['SLN_Data'][self.areas[i]]
            for j in range(n):
                if self.areas[j] in d:
                    # print('{} {}'.format(self.areas[i], self.areas[j]))
                    result[i,j] = d[self.areas[j]]

        result = np.multiply(result, self.get_connectivity_grid())
        result[np.where(result == 0)] = np.nan
        result = result * 100
        return result

    def get_interpolated_FLNe_Schmidt(self):
        n = len(self.areas)
        interpolated_FLNe = np.zeros((n,n))
        for i in range(n):
            d = self.dat['FLN_completed'][self.areas[i]]
            for j in range(n):
                if self.areas[j] in d:
                    interpolated_FLNe[i,j] = d[self.areas[j]]

        result = np.multiply(interpolated_FLNe, self.get_connectivity_grid())

        # make all the rows sum to 1
        for i in range(n):
            total_fraction = np.sum(result[i,:])
            if total_fraction > 0:
                result[i,:] = result[i,:] / total_fraction

        return result

    def compare_markov_and_non_markov_connection_strengths(self):
        n = len(self.areas)
        interpolated_FLNe = np.zeros((n,n))
        for i in range(n):
            d = self.dat['FLN_completed'][self.areas[i]]
            for j in range(n):
                if self.areas[j] in d:
                    interpolated_FLNe[i,j] = d[self.areas[j]]

        grid = self.get_connectivity_grid()

        remapped_areas = ['V1', 'V2', 'V4', 'MSTd', 'DP', 'CITv', 'FEF', '7a', 'STPp', 'STPa', '46']
        markov = []
        non_markov = []
        for i in range(n):
            if self.areas[i] in remapped_areas:
                for j in range(n):
                    if interpolated_FLNe[i][j] > 0:
                        if grid[i][j]:
                            markov.append(interpolated_FLNe[i][j])
                        else:
                            non_markov.append(interpolated_FLNe[i][j])

        print('connection strength ratio: {}'.format(np.mean(non_markov) / np.mean(markov)))

        markov = np.log10(markov)
        non_markov = np.log10(non_markov)
        low = min(np.min(markov), np.min(non_markov))
        high = max(np.max(markov), np.max(non_markov))

        # Remapped connections not in CoCoMac are fairly strong, so we assume they
        # are spurious rather than "not found previously" as in Markov et al. (2014)
        # Compare Markov et al. Figure 9 (mode of NFP strengths is about -4.5, in
        # contrast with about -3 here).
        bins = np.linspace(low, high, 21)
        plt.figure(figsize=(4,3))
        plt.subplot(2,1,1)
        plt.hist(markov, bins)
        plt.ylim((0, 30))
        plt.ylabel('Count')
        plt.title('Remapped connections in CoCoMac')
        plt.subplot(2,1,2)
        plt.hist(non_markov, bins)
        plt.ylim((0, 30))
        plt.ylabel('Count')
        plt.xlabel('Log-FLN')
        plt.title('Remapped connections not in CoCoMac')
        plt.tight_layout()
        plt.show()


def sigmoid(x, centre, gain):
    """
    A sigmoid function, which we use to fit SLN data as a function of log-ratio
    of neuron densities.

    :param x: value(s) of independent variable
    :param centre: value at which the result is 0.5
    :param gain: linearly related to the slope at the centre
    :return: sigmoid function of x with parameters centre and gain
    """
    return 1 / (1 + np.exp(-gain*(x-centre)))


""" 
Data from M. Schmidt, R. Bakker, C. C. Hilgetag, M. Diesmann, and S. J. van Albada, “Multi-scale account of the network structure of 
macaque visual cortex,” Brain Struct. Funct., vol. 223, no. 3, pp. 1409–1435, 2018.
"""

# Note most of these are similar to surface areas in FV91, but some are quite different, e.g. V1 (1484.63 vs. 1120),
# DP (113.83 vs. 50), and STPp (245.48 vs. 120).
S18_surface_area = {
    'V1': 1484.63,
    'V3': 120.57,
    'PO': 75.37,
    'V2': 1193.40,
    'CITv': 114.67,
    'VOT': 70.11,
    'V4': 561.41,
    'DP': 113.83,
    'FST': 61.33,
    'STPp': 245.48,
    'PIP': 106.15,
    'CITd': 57.54,
    'TF': 197.40,
    'PITv': 100.34,
    'LIP': 56.04,
    '46': 185.16,
    'V3A': 96.96,
    'MT': 55.90,
    'FEF': 161.54,
    'AITv': 93.12,
    'MIP': 45.09,
    '7a': 157.34,
    'AITd': 91.59,
    'TH': 44.60,
    'PITd': 145.38,
    'VIP': 85.06,
    'MSTl': 29.19,
    'VP': 130.58,
    'STPa': 78.72,
    'V4t': 28.23,
    'MSTd': 120.57,
    'MDP': 77.49
}

S18_distance = [
    [0,17.9,19.9,14.6,16.8,22.5,23.1,22.9,29,26.8,18.8,21.5,23.7,24.5,29.2,26.3,27.8,32.9,31.6,28.4,38.8,37.7,57.1,29.6,43.8,33.7,28.2,38,44.3,62.9,46.3,30.8],
    [17.9,0,16.1,17.8,18.2,20,20.5,21.2,24.5,24.4,19.8,23.8,24.6,26,30.8,25.8,27.6,28,27.3,24.4,33.4,32.5,53.9,24.8,38.2,28.9,27.6,34.3,39.2,59.5,40.9,26.3],
    [19.9,16.1,0,20.8,19,14.9,15.1,14.6,12.8,20.9,20.1,25.2,25.4,26.9,31.9,25.5,28,16.8,17.8,17.4,22.4,22,48.3,16.3,28.2,20.3,27.4,27.7,30.2,54.1,31.2,19.9],
    [14.6,17.8,20.8,0,8.1,15.9,17,18.5,26.9,19.4,10.6,14.6,15.1,16.9,22,18.1,19.3,30.4,27.4,22.6,36,34.3,50,27.4,41.1,28.8,19.9,31.7,40.2,55.9,43.3,27.5],
    [16.8,18.2,19,8.1,0,12.4,13.4,15.6,23.4,15.4,9.2,15,9.4,16.3,21.2,13.8,14.6,26.8,23.5,18.9,32.7,31,45.8,24.7,38.1,25.6,14.5,28.1,37.1,51.8,40.4,25.1],
    [22.5,20,14.9,15.9,12.4,0,6,11.4,13.7,10,13.2,19.1,16.6,20,23.9,13.7,16,16.3,13.1,8.2,21.6,19.5,37.9,15.8,26.6,14.2,15.5,18.1,25.5,43.9,28.4,17.1],
    [23.1,20.5,15.1,17,13.4,6,0,9.9,12.1,12.1,15.2,21.3,17.8,22.1,26.3,16.4,18.6,14.8,11.6,9.9,21.1,19.2,40.3,17.1,27,15.7,18,20,27,46.4,29,18.9],
    [22.9,21.2,14.6,18.5,15.6,11.4,9.9,0,13.1,17.8,18.6,24.6,20.4,25.6,30.5,21.4,23.3,16.3,14.1,15.7,22.5,20.9,45.8,20.1,28.9,19.2,22.4,25.5,30,51.5,30.3,22.6],
    [29,24.5,12.8,26.9,23.4,13.7,12.1,13.1,0,19.7,24.6,30,28.5,31,36,26.6,29.1,7.4,8.6,13.2,12.9,12.2,42.1,14.4,19.5,12.8,27.8,22.5,21.8,47.7,21.6,18.1],
    [26.8,24.4,20.9,19.4,15.4,10,12.1,17.8,19.7,0,14.5,20.6,17.1,20.2,24.1,11.5,12.4,21.5,18.7,9.1,25.7,23.2,36.4,20.2,30.1,16.5,11,16,27.3,42.4,31.6,20.6],
    [18.8,19.8,20.1,10.6,9.2,13.2,15.2,18.6,24.6,14.5,0,9.5,12.3,9.8,14.2,11.2,13.6,27.6,24.8,18,32.4,30.5,42.9,23.5,36.7,24.6,14.5,27.2,35.3,49,39,22.4],
    [21.5,23.8,25.2,14.6,15,19.1,21.3,24.6,30,20.6,9.5,0,18.9,6.9,10,16,19.5,32.9,30.6,24,37.6,35.9,47.9,28,41.4,29.8,20.9,32.9,40.4,53.3,43.9,26.2],
    [23.7,24.6,25.4,15.1,9.4,16.6,17.8,20.4,28.5,17.1,12.3,18.9,0,18.3,22,11.1,10.1,31.9,28.2,22.7,37.7,35.7,46.2,29.8,42.7,29.6,11.5,30.8,40.9,52,44.7,29.4],
    [24.5,26,26.9,16.9,16.3,20,22.1,25.6,31,20.2,9.8,6.9,18.3,0,6.3,14.6,18.2,34,31.4,24.2,38.8,37,45.4,29.5,42.5,30.6,20,33,41,50.8,44.7,28.2],
    [29.2,30.8,31.9,22,21.2,23.9,26.3,30.5,36,24.1,14.2,10,22,6.3,0,17.4,21.1,38.7,35.9,28.3,43.1,41,46.7,33.7,46.7,34.2,23.2,36.6,44.5,52,48.9,31.4],
    [26.3,25.8,25.5,18.1,13.8,13.7,16.4,21.4,26.6,11.5,11.2,16,11.1,14.6,17.4,0,7.4,28.8,25.7,16.9,33.1,30.7,36.8,25.9,36.5,23.8,9.4,24.3,33.6,42.6,38.1,24.9],
    [27.8,27.6,28,19.3,14.6,16,18.6,23.3,29.1,12.4,13.6,19.5,10.1,18.2,21.1,7.4,0,31.2,28,18.7,35.5,33.1,39,28.9,39.5,26.2,7.1,25.5,36.2,45.1,40.9,28.2],
    [32.9,28,16.8,30.4,26.8,16.3,14.8,16.3,7.4,21.5,27.6,32.9,31.9,34,38.7,28.8,31.2,0,8.3,13.9,9.3,8.3,40.5,14.1,15.4,11.2,29.7,21.6,18.4,45.8,17.2,18.1],
    [31.6,27.3,17.8,27.4,23.5,13.1,11.6,14.1,8.6,18.7,24.8,30.6,28.2,31.4,35.9,25.7,28,8.3,0,12.1,12.7,10.4,39.9,15.8,19.1,10.9,26.5,20.2,20.2,45.4,19.7,19.2],
    [28.4,24.4,17.4,22.6,18.9,8.2,9.9,15.7,13.2,9.1,18,24,22.7,24.2,28.3,16.9,18.7,13.9,12.1,0,17.7,15.1,32.4,14.3,22.2,8.8,17.2,11.9,19.8,38.4,23.9,15.8],
    [38.8,33.4,22.4,36,32.7,21.6,21.1,22.5,12.9,25.7,32.4,37.6,37.7,38.8,43.1,33.1,35.5,9.3,12.7,17.7,0,6.4,38.7,14.6,9.3,11.8,33.9,20.9,13.8,43.2,10.9,18.3],
    [37.7,32.5,22,34.3,31,19.5,19.2,20.9,12.2,23.2,30.5,35.9,35.7,37,41,30.7,33.1,8.3,10.4,15.1,6.4,0,36.5,13.9,10.3,9,31.4,18.5,12.4,41.2,10.7,17.2],
    [57.1,53.9,48.3,50,45.8,37.9,40.3,45.8,42.1,36.4,42.9,47.9,46.2,45.4,46.7,36.8,39,40.5,39.9,32.4,38.7,36.5,0,39.9,36.5,30.5,39.4,33.3,29.3,11.2,35.2,39.8],
    [29.6,24.8,16.3,27.4,24.7,15.8,17.1,20.1,14.4,20.2,23.5,28,29.8,29.5,33.7,25.9,28.9,14.1,15.8,14.3,14.6,13.9,39.9,0,16.4,12.7,27.9,21.7,19,44.7,19.6,9.7],
    [43.8,38.2,28.2,41.1,38.1,26.6,27,28.9,19.5,30.1,36.7,41.4,42.7,42.5,46.7,36.5,39.5,15.4,19.1,22.2,9.3,10.3,36.5,16.4,0,14.3,38.3,21.7,10.7,39.9,7.4,18.5],
    [33.7,28.9,20.3,28.8,25.6,14.2,15.7,19.2,12.8,16.5,24.6,29.8,29.6,30.6,34.2,23.8,26.2,11.2,10.9,8.8,11.8,9,30.5,12.7,14.3,0,24.7,12.4,12.2,36,15.5,14.6],
    [28.2,27.6,27.4,19.9,14.5,15.5,18,22.4,27.8,11,14.5,20.9,11.5,20,23.2,9.4,7.1,29.7,26.5,17.2,33.9,31.4,39.4,27.9,38.3,24.7,0,23.6,35.3,45.4,40,27.6],
    [38,34.3,27.7,31.7,28.1,18.1,20,25.5,22.5,16,27.2,32.9,30.8,33,36.6,24.3,25.5,21.6,20.2,11.9,20.9,18.5,33.3,21.7,21.7,12.4,23.6,0,16,38.4,22.2,23.3],
    [44.3,39.2,30.2,40.2,37.1,25.5,27,30,21.8,27.3,35.3,40.4,40.9,41,44.5,33.6,36.2,18.4,20.2,19.8,13.8,12.4,29.3,19,10.7,12.2,35.3,16,0,33.1,10.2,20.7],
    [62.9,59.5,54.1,55.9,51.8,43.9,46.4,51.5,47.7,42.4,49,53.3,52,50.8,52,42.6,45.1,45.8,45.4,38.4,43.2,41.2,11.2,44.7,39.9,36,45.4,38.4,33.1,0,38.3,44.6],
    [46.3,40.9,31.2,43.3,40.4,28.4,29,30.3,21.6,31.6,39,43.9,44.7,44.7,48.9,38.1,40.9,17.2,19.7,23.9,10.9,10.7,35.2,19.6,7.4,15.5,40,22.2,10.2,38.3,0,21.8],
    [30.8,26.3,19.9,27.5,25.1,17.1,18.9,22.6,18.1,20.6,22.4,26.2,29.4,28.2,31.4,24.9,28.2,18.1,19.2,15.8,18.3,17.2,39.8,9.7,18.5,14.6,27.6,23.3,20.7,44.6,21.8,0]
]

"""
TODO: omit callosal neurons:
J. C. Houzel, M. L. Carvalho, and R. Lent, “Interhemispheric connections between primary visual 
areas: Beyond the midline rule,” Brazilian J. Med. Biol. Res., vol. 35, no. 12, pp. 1441–1453, 2002.

Also account for non-projecting pyramidal cells: 
Gilbert, Charles D., and TORSTEN N. Wiesel. "Clustered intrinsic connections in cat visual cortex." 
Journal of Neuroscience 3.5 (1983): 1116-1133.
"""

"""
This code is vision-centric at the moment. Here are some relevant references for other parts of cortex:
 
Barbas, H., & Rempel-Clower, N. (1997). Cortical structure predicts the pattern of corticocortical 
connections. Cerebral cortex (New York, NY: 1991), 7(7), 635-646.

Dombrowski, S. M., Hilgetag, C. C., & Barbas, H. (2001). Quantitative architecture distinguishes 
prefrontal cortical systems in the rhesus monkey. Cerebral Cortex, 11(10), 975-988.

Dum, R. P., & Strick, P. L. (2005). Motor areas in the frontal lobe: the anatomical substrate for 
the central control of movement. Motor cortex in voluntary movements: a distributed system for 
distributed functions, 3-47. 

G. N. Elston, “Cortical heterogeneity: Implications for visual processing and polysensory integration,” 
J. Neurocytol., vol. 31, no. 3–5 SPEC. ISS., pp. 317–335, 2002.
"""

def data_folder():
    return os.path.dirname(inspect.stack()[0][1]) + '/data_files'


# Histograms of synpapses per functional connection in rat barrel cortex. We average over three
# connections. From:
#
# T. Fares and A. Stepanyants, “Cooperative synapse formation in the neocortex.,”
# Proc. Natl. Acad. Sci. U. S. A., vol. 106, no. 38, pp. 16463–16468, 2009.
#
# Note this non-independent synapse formation is qualitatively consistent with:
#
# N. Kasthuri, K. J. Hayworth, D. R. Berger, R. L. Schalek, J. A. Conchello,
# S. Knowles-Barley, D. Lee, A. Vázquez-Reina, V. Kaynig, T. R. Jones, M. Roberts,
# J. L. Morgan, J. C. Tapia, H. S. Seung, W. G. Roncal, J. T. Vogelstein, R. Burns,
# D. L. Sussman, C. E. Priebe, H. Pfister, and J. W. Lichtman, “Saturated Reconstruction
# of a Volume of Neocortex,” Cell, vol. 162, no. 3, pp. 648–661, 2015.
FS09_synapses_per_connection = [
    [4, 0.5385996409335727], # L4->L2/3
    [5, 0.4631956912028726],
    [4, 0.10626118067978559], # L5->L5
    [5, 0.4754919499105546],
    [6, 0.21037567084078707],
    [7, 0.15885509838998244],
    [8, 0.05259391771019685],
    [2, 0.18107142857142855], # L4->L4
    [3, 0.4553571428571429],
    [4, 0.18214285714285722],
    [5, 0.18214285714285705]
]


# Sincich, L. C., Adams, D. L., & Horton, J. C. (2003). Complete flatmounting of the macaque cerebral
# cortex. Visual neuroscience, 20(6), 663-686.
# (These all vary within a factor of two, but surface area of V1 & V2 well correlated with total; Fig 7)
SAH_area = {
    'V1': 1343,
    'V2': 1012,
    'MT': 73,
    'A1': 88,
    'S1': 284,
    'Hippocampus': 181,
    'Neocortex': 10430
}


# Garcia-Marin, V., Kelly, J. G., & Hawken, M. J. (2017). Major feedforward thalamic input into layer
# 4C of primary visual cortex in primate. Cerebral Cortex, 1-16.
GMKH17_density_per_mm3_V1 = {
    '1': 20700,
    '2/3': 257800,
    '4A': 268800,
    '4B': 173800,
    '4Calpha': 235300,
    '4Cbeta': 410600,
    '4': 259120, # average of sublayers weighted by thickness
    '5': 213000,
    '6': 214500,
    '1-6': 229100
}

GMKH17_density_per_mm2_V1 = {
    '1': 2500,
    '2/3': 143000,
    '4A': 12800,
    '4B': 36000,
    '4Calpha': 35900,
    '4Cbeta': 56500,
    '4': 12800 + 36000 + 35900 + 56500,
    '5': 36200,
    '6': 55500,
    '1-6': 378300
}

GMKH17_thickness_V1 = {
    '1': 0.120772947,
    '2/3': 0.554693561,
    '4A': 0.047619048,
    '4B': 0.207134638,
    '4Calpha': 0.152571186,
    '4Cbeta': 0.137603507,
    '4': 0.047619048 + 0.207134638 + 0.152571186 + 0.137603507,
    '5': 0.169953052,
    '6': 0.258741259,
    '1-6': 1.651243998
}


# O'Kusky, J., & Colonnier, M. (1982). A laminar analysis of the number of neurons, glia, and synapses in the visual
# cortex (area 17) of adult macaque monkeys. Journal of Comparative Neurology, 210(3), 278-290.
OC82_thickness_V1 = { # in mm
    '1': .1229,
    '2/3': .3969,
    '4A': .1271,
    '4B': .2114,
    '4C': .2475,
    '5': .2263,
    '6': .2602
}

OC82_n_neurons_per_mm2_V1 = {
    '1': 600,
    '2/3-1': 20200,
    '2/3-2': 17200,
    '2/3-3': 18700,
    '4A': 21800,
    '4B-1': 12200,
    '4B-2': 9000,
    '4Calpha': 17300,
    '4Cbeta': 30000,
    '5-1': 16000,
    '5-2': 8800,
    '6A': 23400,
    '6B': 6700
}

OC82_n_synapses_per_mm2_V1 = {
    '1': 33300000,
    '2/3-1': 50800000,
    '2/3-2': 51600000,
    '2/3-3': 52100000,
    '4A': 34900000,
    '4B-1': 26300000,
    '4B-2': 25900000,
    '4Calpha': 36900000,
    '4Cbeta': 44000000,
    '5-1': 33600000,
    '5-2': 32100000,
    '6A': 32000000,
    '6B': 25000000
}

# Balaram, P., Young, N. A., & Kaas, J. H. (2014). Histological features of layers and sublayers in
# cortical visual areas V1 and V2 of chimpanzees, macaque monkeys, and humans. Eye and brain, 2014(6 Suppl 1), 5.
# This data was estimated from Figure 4 using WebPlotDigitizer. Laminae labels are Brodman's.
BYK14_thickness_V1 = {
    '1': 0.0828,
    '2': 0.0552,
    '3A': 0.1345,
    '3B': 0.1207,
    '4A': 0.0276,
    '4B': 0.0810,
    '4Calpha': 0.0672,
    '4Cbeta': 0.0828,
    '5A': 0.0379,
    '5B': 0.0603,
    '6A': 0.0638,
    '6B': 0.0776
}

BYK14_thickness_V2 = {
    '1': 0.0931,
    '2/3': 0.0828 + 0.1155 + 0.1879, #sum of 2, 3A, 3B
    '4': 0.1483,
    '5': 0.0414 + 0.0897, # sum of 5A, 5B
    '6': 0.0897 + 0.1552 # sum of 6A, 6B
}

# Shepherd, G. M. (Ed.). (2003). The synaptic organization of the brain, 5th Ed. Oxford University Press.
S03_thickness_V1 = {
    '1': 0.0962,
    '2': 0.1404,
    '3': 0.2334,
    '4A': 0.0394,
    '4B': 0.1277,
    '4Calpha': 0.1105,
    '4Cbeta': 0.1341,
    '5': 0.0882,
    '6': 0.1877
}

S03_thickness_V2 = {
    '1': 0.0989,
    '2/3': 0.1798 + 0.3375, # sum of 2 and 3
    '4': 0.1514,
    '5': 0.0894,
    '6': 0.1431
}

# Binzegger, T., Douglas, R. J., & Martin, K. A. (2004). A quantitative map of the circuit of cat
# primary visual cortex. Journal of Neuroscience, 24(39), 8441-8453.
BDM04_total_area_hemisphere = 399 #mm squared

# This is per hemisphere, extrapolated from V1, extracted from Fig 6A via WebPlotDigitizer
BDM04_N_per_type = {
    'sp1': 25050,
    'sm1': 480962,
    'p2/3': 8252505,
    'b2/3': 964930,
    'db2/3': 572144,
    'axo2/3': 88176,
    'sm2/3': 691383,
    'ss4(L4)': 2900802,
    'ss4(L2/3)': 2900802,
    'p4': 2900802,
    'b4': 1694389,
    'sm4': 480962,
    'p5(L2/3)': 1512024,
    'p5(L5/6)': 389780,
    'b5': 179359,
    'sm5': 242485,
    'p6(L4)': 4296593,
    'p6(L5/6)': 1420842,
    'sm6': 1175351,
    'X/Y': 361723
}

BDM04_targets = [
    'sp1',
    'sm1',
    'p2/3',
    'b2/3',
    'db2/3',
    'axo2/3',
    'sm2/3',
    'ss4(L4)',
    'ss4(L2/3)',
    'p4',
    'b4',
    'sm4',
    'p5(L2/3)',
    'p5(L5/6)',
    'b5',
    'sm5',
    'p6(L4)',
    'p6(L5/6)',
    'sm6'
]

BDM04_sources = [
    'p2/3',
    'b2/3',
    'db2/3',
    'axo2/3',
    'ss4(L4)',
    'ss4(L2/3)',
    'p4',
    'b4',
    'p5(L2/3)',
    'p5(L5/6)',
    'b5',
    'p6(L4)',
    'p6(L5/6)',
    'X/Y',
    'as', #unknown asymmetric
    'sy'
]

BDM04_excitatory_types = {
    '1': ['sp1'],
    '2/3': ['p2/3'],
    '4': ['ss4(L4)', 'ss4(L2/3)', 'p4'],
    '5': ['p5(L2/3)', 'p5(L5/6)'],
    '6': ['p6(L4)', 'p6(L5/6)'],
    'thalamocortical': ['X/Y'],
    'extrinsic': ['as'],
    'interlaminar': ['sp1', 'p2/3', 'ss4(L4)', 'ss4(L2/3)', 'p4', 'p5(L2/3)', 'p5(L5/6)', 'p6(L4)', 'p6(L5/6)'],
    'all': ['sp1', 'p2/3', 'ss4(L4)', 'ss4(L2/3)', 'p4', 'p5(L2/3)', 'p5(L5/6)', 'p6(L4)', 'p6(L5/6)', 'X/Y', 'as']
}

# Extracted from Fig 9A via WebPlotDigitizer; needed to check import of tables in supplementary material

BDM04_synapses_per_neuron_L1 = {
    'sp1': 13005,
    'sm1': 8808,
    'p2/3': 1241,
    'b2/3': 532,
    'db2/3': 1300,
    'p4': 768,
    'p5(L2/3)': 236,
    'p5(L5/6)': 5616,
    'p6(L5/6)': 177
}

BDM04_synapses_per_neuron_L23 = {
    'p2/3': 5789,
    'b2/3': 3256,
    'db2/3': 2050,
    'axo2/3': 3015,
    'sm2/3': 3015,
    'ss4(L2/3)': 60,
    'p4': 784,
    'b4': 60,
    'p5(L2/3)': 302,
    'p5(L5/6)': 1266,
    'p6(L4)': 60,
    'p6(L5/6)': 241,
}

BDM04_synapses_per_neuron_L4 = {
    'ss4(L4)': 5789,
    'ss4(L2/3)': 4764,
    'p4': 4764,
    'b4': 3015,
    'sm4': 3618,
    'p5(L2/3)': 302,
    'p5(L5/6)': 905,
    'p6(L4)': 1809,
    'p6(L5/6)': 241
}

BDM04_synapses_per_neuron_L5 = {
    'p4': 294,
    'p5(L2/3)': 4118,
    'p5(L5/6)': 4882,
    'b5': 2941,
    'sm5': 2941,
    'p6(L4)': 1000,
    'p6(L5/6)': 294
}

BDM04_synapses_per_neuron_L6 = {
    'p5(L5/6)': 235,
    'p6(L4)': 3294,
    'p6(L5/6)': 5647,
    'sm6': 3176
}

# From Figure 4 of Felleman, D. J., & Van Essen, D. C. (1991). Distributed hierarchical
# processing in the primate cerebral cortex. Cerebral cortex (New York, NY: 1991), 1(1),
# 1-47.
FV91_hierarchy = {
    'V1': 1,
    'V2': 2,
    'V3': 3, 'VP': 3,
    'PIP': 4, 'V3A': 4,
    'MDP': 5, 'MIP': 5, 'PO': 5, 'MT': 5, 'V4t': 5, 'V4': 5,
    'DP': 6, 'VOT': 6,
    'VIP': 7, 'LIP': 7, 'MSTd': 7, 'MSTl': 7, 'FST': 7, 'PITd': 7, 'PITv': 7,
    '7b': 8, '7a': 8, 'FEF': 8, 'STPp': 8, 'CITd': 8, 'CITv': 8,
    'STPa': 9, 'AITd': 9, 'AITv': 9,
    '36': 10, '46': 10, 'TF': 10, 'TH': 10
}


class E07:
    """
    Data on spine counts from G. N. Elston, “Specialization of the Neocortical Pyramidal
    Cell during Primate Evolution,” Evol. Nerv. Syst., pp. 191–242, 2007.
    """
    def __init__(self):

        # From figures 8 and 11
        self.layer_3_basal_spine_count = {
            'V1': 812,
            'V2': 1326,
            'V4': 2588,
            'TEO': 5023,
            'TE': 7385,
            '3b': 3062,
            '4': 4598,
            '5': 4735,
            '6': 8318,
            '7': 6892
        }

        # Mappings from area names used in Elston to corresponding areas in Yerkes atlas
        self.yerkes_mappings = {
            'TE': ['TEad', 'TEa/ma', 'TEa/mp', 'TEav', 'TEpd', 'TEpv'],
            '7': ['7A', '7B', '7m', '7op'],
            '3b': ['3'],
            '4': ['F1'],
            '6': ['F3', 'F6']  # see Luppino & Rizzolati (2000)
        }

        self.FV91_mappings = {
            'TE': ['PITd', 'PITv', 'CITd', 'CITv', 'AITd', 'AITv'],
            'TEO': ['VOT'],
            '7': ['7a'],
            '3b': [],
            '4': [],
            '5': [],
            '6': []
        }

        self._V1_ind = areas_FV91.index('V1')
        self._fit()

    def _fit(self):

        distances = []
        spine_counts = []
        for key in self.layer_3_basal_spine_count.keys():
            if key in areas_FV91:
                areas = [key]
            else:
                areas = self.FV91_mappings[key]

            if areas:
                distance = np.mean([self.get_distance(area) for area in areas])
                distances.append(distance)
                spine_counts.append(self.layer_3_basal_spine_count[key])

        p = np.polyfit(distances, spine_counts, 1)
        self.intercept = p[1]
        self.slope = p[0]

    def get_spine_count(self, area):
        if area in self.layer_3_basal_spine_count.keys():
            return self.layer_3_basal_spine_count[area]
        else:
            for key in self.FV91_mappings.keys():
                if area in self.FV91_mappings[key]:
                    return self.layer_3_basal_spine_count[key]

            if area in areas_FV91:
                return self.intercept + self.slope * self.get_distance(area)
            else:
                return None

    def get_distance(self, area):
        """
        :param area: A FV91 area
        :return: Distance to this area from V1 (mean through white matter)
        """
        return S18_distance[self._V1_ind][areas_FV91.index(area)]

    def plot(self):
        import matplotlib.pyplot as plt

        interpolated_distances = []
        interpolated_spine_counts = []
        measured_distances = []
        measured_spine_counts = []

        for area in areas_FV91:
            measured = False
            if area in self.layer_3_basal_spine_count.keys():
                measured = True
            else:
                for key in self.FV91_mappings.keys():
                    if area in self.FV91_mappings[key]:
                        measured = True

            if measured:
                measured_distances.append(self.get_distance(area))
                measured_spine_counts.append(self.get_spine_count(area))
            else:
                interpolated_distances.append(self.get_distance(area))
                interpolated_spine_counts.append(self.get_spine_count(area))

        plt.figure(figsize=(3.5, 2.5))
        plt.plot(measured_distances, measured_spine_counts, 'bo')
        plt.plot(interpolated_distances, interpolated_spine_counts, 'k.')
        plt.plot([0, 65], self.intercept + [0, self.slope*65], 'k')
        plt.xlabel('Distance from V1')
        plt.ylabel('Basal spine count')
        plt.tight_layout()
        plt.savefig('../generated-files/figures/spine-count.eps')
        plt.show()


class SchmidtData:
    """
    Data on mean inbound synapses per neuron. Only excitatory cells are considered, based on the
    rationale in:

    Parisien, C., Anderson, C. H., & Eliasmith, C. (2008). Solving the problem of negative
    synaptic weights in cortical models. Neural computation, 20(6), 1473-1494.

    Tripp, B., & Eliasmith, C. (2016). Function approximation in inhibitory networks.
    Neural Networks, 77, 95-106.
    """

    def __init__(self):
        with open('./data_files/schmidt/default_Data_Model_.json') as file:
            self.data = json.load(file)

    def neuron_numbers(self, area, layer):
        return self.data['realistic_neuron_numbers'][area][self._adapt_layer_name(layer)]

    def interarea_synapses_per_neuron(self, area, target_layer):
        """
        :param area: name of area
        :param target_layer: name of layer that contains cell bodies of target cells
        :return: estimate of mean # excitatory synapses per neuron from other cortical areas
        """
        return self._synapses_per_neuron(area, target_layer, include_interlaminar=False)[0]

    def interlaminar_synapses_per_neuron(self, area, source_layer, target_layer):
        """
        :param area: name of area
        :param source_layer: name of layer that contains cell bodies of source cells
        :param target_layer: name of layer that contains cell bodies of target cells
        :return: estimate of mean # excitatory synapses per neuron from within same cortical area
            and given source layer
        """
        patch_factor = self._get_patch_factor(area, target_layer)

        source_layer = self._adapt_layer_name(source_layer)
        target_layer = self._adapt_layer_name(target_layer)

        synapses = self.data['synapses'][area][target_layer][area][source_layer]
        neuron_number = self.data['neuron_numbers'][area][target_layer]

        return synapses / neuron_number * patch_factor

    def _get_patch_factor(self, area, target_layer):
        """
        Estimates the ratio of interlaminar excitatory synapses to interlaminar excitatory synapses
        within a 1mm^2 patch. The latter is the number estimated by Schmidt et al., but we want
        the former. We assume all Schmidt et al.'s "external" synapses are from the same area but
        outside the patch. This neglects synapses from areas outside the visual cortex (which Schmidt
        et al. also count as "external" to their model) but these are probably a small minority
        (Markov et al. 2011).
        """
        excitatory, inhibitory = self._synapses_per_neuron(area, target_layer, include_interarea=False)

        excitatory_fraction = excitatory / (excitatory+inhibitory)
        external = self._synapses_per_neuron_external(area, target_layer)
        excitatory_full = excitatory + excitatory_fraction*external

        return excitatory_full / excitatory

    def _synapses_per_neuron(self, area, target_layer, include_interlaminar=True, include_interarea=True):
        target = self.data['synapses'][area][self._adapt_layer_name(target_layer)]

        excitatory = 0
        inhibitory = 0
        for source_area in target.keys():
            interlaminar = (source_area == area)
            if (interlaminar and include_interlaminar) or (not interlaminar and include_interarea):
                for source_layer in target[source_area].keys():
                    n = target[source_area][source_layer]
                    if source_layer in ['23E', '4E', '5E', '6E']:
                        excitatory += n
                    elif source_layer in ['23I', '4I', '5I', '6I']:
                        inhibitory += n

        n = self._num_neurons(area, target_layer)

        return excitatory/n, inhibitory/n

    def _adapt_layer_name(self, layer):
        return '{}E'.format(layer.replace('/', ''))

    def _num_neurons(self, area, target_layer):
        return self.data['neuron_numbers'][area][self._adapt_layer_name(target_layer)]

    def _synapses_per_neuron_external(self, area, target_layer):
        target = self.data['synapses'][area][self._adapt_layer_name(target_layer)]
        return target['external']['external'] / self._num_neurons(area, target_layer)



"""
Sources for receptive field sizes: 

Freeman, J., & Simoncelli, E. P. (2011). Metamers of the ventral stream. Nature neuroscience, 
14(9), 1195-1201.

Romero, M. C., & Janssen, P. (2016). Receptive field properties of neurons in the macaque 
anterior intraparietal area. Journal of neurophysiology, 115(3), 1542-1555.

Mendoza-Halliday, D., Torres, S., & Martinez-Trujillo, J. C. (2014). Sharp emergence of 
feature-selective sustained activity along the dorsal visual pathway. Nature neuroscience, 
17(9), 1255-1262.

Kravitz, D. J., Saleem, K. S., Baker, C. I., Ungerleider, L. G., & Mishkin, M. (2013). 
The ventral visual pathway: an expanded neural framework for the processing of object quality. 
Trends in cognitive sciences, 17(1), 26-49.

Op De Beeck, H., & Vogels, R. (2000). Spatial sensitivity of macaque inferior temporal neurons. 
Journal of Comparative Neurology, 426(4), 505-518.
"""
RF_diameter_5_degrees_eccentricity = {
    'V1': 1.3, # Gattass et al. (1981)
    'V2': 2.2, # Gattass et al. (1981)
    'V3': 2.8, # mean of Gattass et al. (1988) and Felleman & Van Essen (1987)
    'V4': 4.8, # mean of Gattass et al. (1988) and Boussaoud et al. (1991)
    'MT': 4.2, # mean of Komatsu et al. (1988) and Maunsell et al. (1987)
    'PO': 7.7, # Galletti et al. (1999)
    'MSTd': 16.0, # Komatsu et al. (1988)
    'AIP': None, # but see Romero & Janssen (2016)
    'PITd': 8.9, # Boussaoud et al. (1991)
    'PITv': 8.9, # Boussaoud et al. (1991)
    'CITd': 38.5, # for TE generally in 0-10deg range, from Boussaoud et al. (1991), but sites look fairly caudal (Fig 4)
    'CITv': 38.5
}


def get_RF_size(area):
    result = None
    if area in RF_diameter_5_degrees_eccentricity.keys():
        result = RF_diameter_5_degrees_eccentricity[area]
    return result


class CoCoMac:
    """
    Inter-area connectivity data from the CoCoMac database,
    R. Bakker, T. Wachtler, and M. Diesmann, “CoCoMac 2.0 and the future of tract-tracing
    databases.,” Front. Neuroinform., vol. 6, no. December, p. 30, Jan. 2012.
    Includes categorical connection strength codes, by source and target layer, where
    available.
    """
    def __init__(self):
        folder = data_folder()
        with open(folder + '/cocomac/connectivity_matrix_layers.json') as file:
            self.layers = json.load(file)

        with open(folder + '/cocomac/connectivity_matrix_densities.json') as file:
            self.densities = json.load(file)

        with open(folder + '/cocomac/FV91_to_M132.json') as file:
            self.FV91_to_M132 = json.load(file)

        with open(folder + '/cocomac/M132_to_FV91.json') as file:
            self.M132_to_FV91 = json.load(file)


    @staticmethod
    def _guess_missing_layers(source_area, target_area, layers):
        if layers[0] is None:
            layers[0] = '??????'
        if layers[1] is None:
            layers[1] = '??????'

        source_level = FV91_hierarchy[source_area]
        target_level = FV91_hierarchy[target_area]

        # lacking data we will guess the simpler of typical alternatives from FV91 Figure 3
        if source_level < target_level:
            guess = ['0XX000', '000X00']
        elif source_level > target_level:
            guess = ['0000X0', 'X0000X']
        else:
            guess = ['0XX000', 'XXXXXX']

        layers[0] = list(layers[0])
        layers[1] = list(layers[1])
        for i in range(6):
            if layers[0][i] == '?':
                layers[0][i] = guess[0][i]
            if layers[1][i] == '?':
                layers[1][i] = guess[1][i]
        layers[0] = ''.join(layers[0])
        layers[1] = ''.join(layers[1])

        return layers

    @staticmethod
    def _map_axon_termination_layers_to_cell_layers(layers):
        """
        :param layers: List of six boolean values corresponding to presence of axon terminals in layers 1-6.
        :return: List of five boolean values corresponding to estimated presence of connections to cells in
            layers 1, 2/3, 4, 5, 6. These estimates are made based on tables in Binzegger et al. (2004)
            supplementary material, which breaks down synapses in each layer by target cell type
            and cell-body layer. If >5% of unclassified asymmetric synapses in layer A are onto
            excitatory cells in layer B, we map a True in layer-A input to a True in layer-B return value.

            The mappings are:  axon terminals in layer       suggests synapses onto cells in layer
                                1                               2/3, 4, 5
                                2/3                             2/3
                                4                               4, 6
                                5                               5, 6
                                6                               6

            There are actually no unclassified asymmetric synapses in layer 2/3 according to Binzegger et al.,
            but they studied V1. According to Felleman & Van Essen (1991) we should expect these mainly in
            areas with lateral inter-area connections, which are absent from V1. Here we assume such synapses
            are onto layer 2/3 cells.

            We ignore inhibitory neurons (as we do throughout this project) due to
            reasoning in Parisien et al.

            Code for calculating the fractions is in _print_fraction_asymmetric(), below.
        """
        map = [
            ['2/3', '4', '5'],
            ['2/3'],
            ['2/3'],
            ['4', '6'],
            ['5', '6'],
            ['6']
        ]

        result = []
        for i in range(6):
            if layers[i]:
                result.extend(map[i])

        return sorted(list(set(result)))

    def get_source_areas(self, target_area):
        # http://cocomac.g-node.org/services/connectivity_matrix.php?dbdate=20141022&AP=AxonalProjections_FV91&square=1&merge=max&format=json&cite=1

        source_areas = []
        for key in self.densities.keys():
            targets = self.densities[key]
            # print(targets)
            target_name = 'FV91-{}'.format(target_area)
            if target_name in targets and targets[target_name] != 0:
                source_area = key[5:]
                # print('adding {} density {}'.format(source_area, targets[target_name]))
                source_areas.append(source_area)

        return source_areas

    def get_connection_details(self, source_area, target_area):
        """
        :param source_area: cortical area from which connection originates
        :param target_area: cortical area in which connection terminates
        :return: if a connection between the areas exists in CoCoMac, a dict with keys
            source_layers and target_layers, where each is a list of six boolean values
            indicating whether layers 1-6 participate in the connection at each end.
            Missing information is filled in with typical values according to the areas
            positions in the visual hierarchy, from Felleman, D. J., & Van Essen, D. C.
            (1991). Distributed hierarchical processing in the primate cerebral cortex.
            Cerebral cortex (New York, NY: 1991), 1(1), 1-47. If no connection exists
            between given areas in CoCoMac, None is returned.
        """
        source_area = CoCoMac._map_M14_to_FV91(source_area)
        target_area = CoCoMac._map_M14_to_FV91(target_area)

        layers = self.layers['FV91-{}'.format(source_area)]['FV91-{}'.format(target_area)]

        if layers[0] is None and layers[1] is None:
            return None
        else:
            layers = CoCoMac._guess_missing_layers(source_area, target_area, layers)

            target_present = []
            source_present = []
            for i in range(6):
                source_present.append(False if layers[0][i] == '0' else True)
                target_present.append(False if layers[1][i] == '0' else True)
            return {'source_layers': source_present, 'target_layers': target_present}

    def get_M132_to_FV91(self, area):
        """
        :param area: name of area in M132 atlas
        :return: names of overlapping areas in FV91 atlas with % overlap
        """
        M132_label = '{}_M132'.format(area)
        if M132_label in self.M132_to_FV91.keys():
            result = dict()
            overlaps = self.M132_to_FV91[M132_label]
            for key in overlaps.keys():
                if key.startswith('FVE.'):
                    result[key[4:]] = overlaps[key]
            return result
        else:
            return None


class Markov:
    """
    Data from Markov et al. (2014) Journal of Comparative Neurology and Markov et al. (2014) Cerebral Cortex.
    """
    # TODO: docs, explain averaging over hemispheres

    def __init__(self):
        [self.source_FLN, self.target_FLN, self.fraction_FLN] = _read_fraction_labelled_neurons_extrinsic()
        [self.source_SLN, self.target_SLN, self.percent_SLN] = _read_supragranular_layers_percent()

    def is_injection_site(self, target):
        return target in self.target_FLN

    def get_sources(self, target):
        result = []
        for i in range(len(self.source_FLN)):
            if self.target_FLN[i] == target:
                result.append(self.source_FLN[i])
        return list(set(result))

    def get_FLNe(self, source, target):
        result = []
        for i in range(len(self.source_FLN)):
            if self.source_FLN[i] == source and self.target_FLN[i] == target:
                result.append(self.fraction_FLN[i])

        if result:
            return np.mean(result)
        else:
            return None

    def get_SLN(self, source, target):
        result = []
        for i in range(len(self.source_SLN)):
            if self.source_SLN[i] == source and self.target_SLN[i] == target:
                result.append(self.percent_SLN[i])

        if result:
            return np.mean(result)
        else:
            return 65.42  # mean for ascending connections


def _read_fraction_labelled_neurons_extrinsic():
    sources = []
    targets = []
    fractions = []

    with open(data_folder() + '/markov/Cercor_2012 Table.csv') as csvfile:
        r = csv.reader(csvfile)
        header_line = True
        for row in r:
            if header_line:
                header_line = False
            else:
                # print('{} to {}: {}'.format(row[2], row[3], row[4]))
                sources.append(row[2])
                targets.append(row[3])
                fractions.append(float(row[4]))

    return sources, targets, fractions


def _read_supragranular_layers_percent():
    sources = []
    targets = []
    percents = []
    distances = []

    with open(data_folder() + '/markov/JCN_2013 Table.csv') as csvfile:
        r = csv.reader(csvfile)
        header_line = True
        for row in r:
            if header_line:
                header_line = False
            elif row[2] != 'NA' and len(row[2]) > 0:
                sources.append(row[1])
                targets.append(row[0])
                percents.append(float(row[2]))
                distances.append(float(row[3]))

    return sources, targets, percents


def check_data():
    """
    Some data_files files must be downloaded separately from this repository. This function checks whether
    they are present in the expected form.
    """
    pass


def get_layers(area):
    # Schmidt et al. (2018) say that TH lacks L4 but don't give a reference. However
    # Felleman & Van Essen (1991) say that several connections to TH terminate on L4
    # (F pattern in their Table 5).

    if area == 'V1':
        # return ['1', '2/3', '3B', '4A', '4B', '4Calpha', '4Cbeta', '5', '6']
        return ['1', '2/3', '4A', '4B', '4Calpha', '4Cbeta', '5', '6']
    else:
        return ['1', '2/3', '4', '5', '6']


def _get_thickness_V1(layer):
    """
    :param layer: Layer name
    :return: Thickness in mm. This is taken from population averages in O'Kusky & Colonnier (1982),
        but we subdivide some layers further. The subdivision fractions are based on a single example
        in Balaram et al. (2014)
    """
    thickness = None

    if layer == '2/3':
        t23 = OC82_thickness_V1['2/3']
        t2 = BYK14_thickness_V1['2']
        t3A = BYK14_thickness_V1['3A']
        t3B = BYK14_thickness_V1['3B']
        thickness = t23 * (t2+t3A) / (t2+t3A+t3B) # omit 3B thickness
    elif layer == '3B':
        t23 = OC82_thickness_V1['2/3']
        t2 = BYK14_thickness_V1['2']
        t3A = BYK14_thickness_V1['3A']
        t3B = BYK14_thickness_V1['3B']
        thickness = t23 * (t3B) / (t2+t3A+t3B)
    elif layer == '4Calpha':
        t4C = OC82_thickness_V1['4C']
        t4Calpha = BYK14_thickness_V1['4Calpha']
        t4Cbeta = BYK14_thickness_V1['4Cbeta']
        thickness = t4C * t4Calpha / (t4Calpha+t4Cbeta)
    elif layer == '4Cbeta':
        t4C = OC82_thickness_V1['4C']
        t4Calpha = BYK14_thickness_V1['4Calpha']
        t4Cbeta = BYK14_thickness_V1['4Cbeta']
        thickness = t4C * t4Cbeta / (t4Calpha+t4Cbeta)
    else:
        thickness = OC82_thickness_V1[layer]

    return thickness


def _total_thickness(thickness):
    total = 0
    for key in thickness.keys():
        total = total + thickness[key]
    return total


def markov_FLNe_sums():
    """
    Prints information about total FLNe to visual areas from other visual areas, from
    Markov data.
    """
    visual_areas = ['7A', '7B', '7m', '7op', 'AIP', 'DP', 'FST', 'IPa', 'LIP', 'MIP', 'MST', 'MT', 'PGa', 'PIP', 'Pro.St', 'STPc', 'STPi', 'STPr', 'TEad', 'TEa/ma', 'TEa/mp', 'TEav', 'TEO', 'TEOm', 'TEpd', 'TEpv', 'TH/TF', 'TPt', 'V1', 'V2', 'V3', 'V3A', 'V4', 'V4t', 'V6', 'V6A', 'VIP']

    markov = Markov()
    totals = []
    for target in visual_areas:
        if markov.is_injection_site(target):
            total = 0
            for source in areas_M132:
                FLNe = markov.get_FLNe(source, target)
                if FLNe is not None and source in visual_areas:
                    total += FLNe
            totals.append(total)
            if total < .7:
                print(target)
    print(np.mean(totals))
    print(np.std(totals))
    print(np.min(totals))


if __name__ == '__main__':
    pass
    # markov_FLNe_sums()

    # data = Data()
    # iac = InterAreaConnections()
    # iac.compare_markov_and_non_markov_connection_strengths()

    # area = 'V2'
    # layer = '4'
    # sd = SchmidtData()
    # a = sd.interarea_synapses_per_neuron(area, layer)
    # b = sd.interlaminar_synapses_per_neuron(area, '2/3', layer)
    # c = sd.interlaminar_synapses_per_neuron(area, '4', layer)
    # d = sd.interlaminar_synapses_per_neuron(area, '5', layer)
    # e = sd.interlaminar_synapses_per_neuron(area, '6', layer)
    # print('Schmidt: {} {} {} {} {}'.format(a, b, c, d, e))
    # print('Schmidt total: {}'.format(b + c + d + e))

    # import matplotlib.pyplot as plt
    # data = Data()
    # print(data.synapses_per_connection)

    # n1 = []
    # n2 = []
    # for area in data.get_areas():
    #     print(area)
    #     if area != 'V1':
    #         an1 = 0
    #         an2 = 0
    #         for layer in ['2/3', '4', '5', '6']:
    #             an1 += data.get_num_neurons(area, layer)
    #             an2 += data.get_neuron_numbers_Schmidt(area, layer)
    #             n1.append(data.get_num_neurons(area, layer))
    #             n2.append(data.get_neuron_numbers_Schmidt(area, layer))
    #         # n1.append(an1)
    #         # n2.append(an2)
    # print(np.corrcoef(n1, n2))
    # plt.scatter(n1, n2)
    # plt.show()

    # sum1 = []
    # sum2 = []
    # for area in data.get_areas():
    #     sum1.append(S18_density[area][1] * S18_surface_area[area])
    #     s2 = 0
    #     for layer in data.get_layers(area):
    #         s2 += data.get_num_neurons(area, layer)
    #     sum2.append(s2)
    #
    # print(np.corrcoef(sum1, sum2))
    # plt.scatter(sum1, sum2)
    # plt.show()


    # #TODO: move to unit test
    # for area in areas_FV91:
    #     mm2 = S18_surface_area[area]
    #     n = [data.get_num_neurons(area, layer) for layer in ['1', '2/3', '4', '5', '6']]
    #     d = [x/mm2 for x in n]
    #     print('{} should be {}'.format(np.sum(d)/.75, S18_density[area][1]))
    #     # print('{}: {}'.format(area, d))

    # for area in areas_FV91:
    #     target_layer = '2/3'
    #     inputs = [data.get_inputs_per_neuron(area, source_layer, target_layer) for source_layer in ['1', '2/3', '4', '5', '6']]
    #     print(inputs)

    # for area in areas_FV91:
    #     print(data.get_inputs_per_neuron(area, 'extrinsic', '4'))

    # print(data.get_areas())
    # print(data.get_SLN(data.get_areas()[0], data.get_areas()[1]))
    # print(data.get_FLNe(data.get_areas()[0], data.get_areas()[1]))
    # print(data.get_num_neurons(data.get_areas()[1], '2/3'))
    # print(data.get_layers('V1'))
    # print(data.get_layers('V2'))
    # print(data.get_extrinsic_inputs('V2', '4'))
    # print(data.get_inputs_per_neuron('V2', '4', '2/3'))
    # print(data.get_inputs_per_neuron('V2', '2/3', '5'))
    # print(data.get_receptive_field_size('V2'))
    # print(data.get_source_areas('V2', feedforward_only=True))

    # densities = []
    # distances = []
    # for i in range(len(areas_FV91)):
    #     density = iac._get_density(areas_FV91[i])
    #     if density:
    #         densities.append(density)
    #         distances.append(S18_distance[0][i])
    #
    # interpolated_distances = []
    # interpolated_densities = []
    # for i in range(len(areas_FV91)):
    #     interpolated_densities.append(iac._get_interpolated_density(areas_FV91[i]))
    #     interpolated_distances.append(S18_distance[0][i])
    #
    # densities = np.array(densities)
    # distances = np.array(distances)
    # near_V1 = np.where(distances <= 35)
    # far_V1 = np.where(distances > 35)
    #
    # near_coeffs = np.polyfit(distances[near_V1], densities[near_V1], 1)
    # far_coeffs = np.polyfit(distances[far_V1], densities[far_V1], 1)
    # plt.scatter(interpolated_distances, interpolated_densities)
    # plt.scatter(distances, densities)
    # print(near_coeffs)
    # print(far_coeffs)
    # print(np.mean(densities[far_V1]))
    # plt.plot([0, 30], near_coeffs[1] + [0, near_coeffs[0]*30], 'r')
    # plt.plot([30, 60], [np.mean(densities[far_V1]), np.mean(densities[far_V1])])
    # plt.show()

    # iac = InterAreaConnections()
    # FLNe = iac.get_interpolated_FLNe()
    # FLNe2 = iac.get_interpolated_FLNe_Schmidt()
    #
    # # print(np.sum(FLNe, axis=1))
    # # print(np.sum(FLNe2, axis=1))
    #
    # plt.figure()
    # plt.subplot(211)
    # # SLN = iac.get_interpolated_SLN()
    # # plt.imshow(SLN, vmin=0, vmax=100)
    # # plt.imshow(np.log10(FLNe))
    # plt.imshow(np.log10(FLNe), vmin=np.log10(.000001), vmax=0)
    # plt.subplot(212)
    # # SLN2 = iac.get_interpolated_SLN_Schmidt()
    # # plt.imshow(SLN2, vmin=0, vmax=100)
    # # plt.imshow(np.log10(FLNe2))
    # plt.imshow(np.log10(FLNe2), vmin=np.log10(.000001), vmax=0)
    # plt.show()
    # #
    # # plt.scatter(FLNe.flatten(), FLNe2.flatten())
    # # plt.show()



    # iac = InterAreaConnections()
    # plt.imshow(iac.get_connectivity_grid())
    # plt.show()
    #
    # SLN = iac.get_interpolated_SLN()
    # fig, ax = plt.subplots()
    # im = ax.imshow(SLN, vmin=0, vmax=100)
    # ax.set_xticks(np.arange(len(data.get_areas())))
    # ax.set_yticks(np.arange(len(data.get_areas())))
    # ax.set_xticklabels(data.get_areas(), fontsize=7)
    # ax.set_yticklabels(data.get_areas(), fontsize=7)
    # plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va='center',
    #          rotation_mode="anchor")
    # cbar = ax.figure.colorbar(im, ax=ax, shrink=.5)
    # cbar.ax.set_ylabel(r'SLN', rotation=-90, va="bottom", fontsize=8)
    # cbar.ax.tick_params(labelsize=8)
    # plt.xlabel('Source area')
    # plt.ylabel('Target area')
    # plt.show()
    #
    # FLNe = iac.get_interpolated_FLNe()
    # fig, ax = plt.subplots()
    # im = ax.imshow(np.log10(FLNe), vmin=np.log10(.000001), vmax=0)
    # ax.set_xticks(np.arange(len(data.get_areas())))
    # ax.set_yticks(np.arange(len(data.get_areas())))
    # ax.set_xticklabels(data.get_areas(), fontsize=7)
    # ax.set_yticklabels(data.get_areas(), fontsize=7)
    # plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va='center',
    #          rotation_mode="anchor")
    # cbar = ax.figure.colorbar(im, ax=ax, shrink=.5)
    # cbar.ax.set_ylabel(r'log$_{10}$(FLNe)', rotation=-90, va="bottom", fontsize=8)
    # cbar.ax.tick_params(labelsize=8)
    # plt.xlabel('Source area')
    # plt.ylabel('Target area')
    # plt.show()

    # logFLNe = np.log10(FLNe).flatten()
    # dist = np.array(S18_distance).flatten()
    # ind = np.where(np.logical_and(dist > 0, np.isfinite(logFLNe)))
    # logFLNe = logFLNe[ind]
    # dist = dist[ind]
    # plt.scatter(dist, logFLNe)
    # print(np.max(dist))
    # print(np.max(logFLNe))
    # coeffs = np.polyfit(dist, logFLNe, 1)
    # plt.plot([0, 60], coeffs[1] + [0, 60*coeffs[0]])
    # plt.show()

    # print(data.get_areas())
    # grid = get_connectivity_grid(data.get_areas(), data.cocomac)
    # print(grid)
    # import matplotlib.pyplot as plt
    # plt.imshow(grid)
    # plt.show()

    # print(data.cocomac.M132_to_FV91)
    # print(data.cocomac.get_M132_to_FV91('V1'))

    # synapses_per_neuron('MT', '4', '2/3')

    # e07 = E07()
    # e07.plot()

    # get_centre(self, area):

    # data = OC82_thickness_V1
    # total = 0
    # for key in data.keys():
    #     total = total + data[key]
    # print(total)
    #
    # print(_total_thickness(OC82_thickness_V1))
    #
    # print('V1 thicknesses')
    # for layer in get_layers('V1'):
    #     print(_get_thickness_V1(layer))
    #
    # print('V2 thicknesses')
    # for layer in get_layers('V2'):
    #     print(layer + ': ' + str(_get_thickness_V2(layer)))

    # for layer in get_layers('V1'):
    #     print(_get_neurons_per_mm2_V1(layer))

    # for layer in get_layers('V2'):
    #     print(_get_neurons_per_mm2_V2(layer))

    # import nibabel
    # # foo = nibabel.load('/Users/bptripp/Downloads/Donahue_et_al_2016_Journal_of_Neuroscience_W336/spec/MacaqueYerkes19.R.midthickness.32k_fs_LR.surf.gii')
    # foo = nibabel.load('/Users/bptripp/Downloads/Donahue_et_al_2016_Journal_of_Neuroscience_W336/spec/MacaqueYerkes19.R.very_inflated.32k_fs_LR.surf.gii')
    # # foo.print_summary()
    # bar = foo.get_arrays_from_intent('NIFTI_INTENT_POINTSET')
    # egg = foo.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')
    # # print(egg[0].metadata)
    # print('points and triangles:')
    # print(bar[0].data.shape)
    # # print(max(egg[0].data.flatten()))
    # print(egg[0].data.shape)
    # # # print(dir(egg[0]))
    # # # print(egg[0].darray)
    #
    # y19 = Yerkes19()
    # points_MT = y19.get_points_in_area('MT')
    # points_V1 = y19.get_points_in_area('V1')
    # points_V2 = y19.get_points_in_area('V2')
    # points_TEO = y19.get_points_in_area('TEO')
    #
    #
    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # points = y19.very_inflated_points[::10,:]
    # ax.scatter(points[:,0], points[:,1], points[:,2], c='k', marker='.')
    # ax.scatter(points_MT[:,0], points_MT[:,1], points_MT[:,2], c='b', marker='o')
    # ax.scatter(points_TEO[:,0], points_TEO[:,1], points_TEO[:,2], c='r', marker='o')
    # # ax.scatter(points_V1[:,0], points_V1[:,1]-2, points_V1[:,2], c='r', marker='.')
    # # ax.scatter(points_V2[:,0], points_V2[:,1]+2, points_V2[:,2], c='g', marker='.')
    # plt.show()

    # print(y19.get_surface_area_total())

    # y19._assign_triangles()
    # print(y19.get_surface_area('V1'))
    # print(y19.get_surface_area('V2'))
    # print(y19.get_surface_area('MT'))
    #
    # print(get_layers('V1'))
    # print(get_num_neurons('V1', '2/3'))
    # print(get_num_neurons('V1', '3B'))
    # print(get_num_neurons('V1', '4A'))
    # print(get_num_neurons('V1', '4B'))
    # print(get_num_neurons('V1', '4Calpha'))
    # print(get_num_neurons('V1', '4Cbeta'))
    # print(get_num_neurons('V1', '5'))
    # print(get_num_neurons('V2', '2/3'))
    #
    # print(get_areas())

    # print(_get_neurons_per_mm2_V1('2/3'))
    # print(_get_neurons_per_mm2_V1('3B'))
    # print(_get_neurons_per_mm2_V1('4A'))
    # print(_get_neurons_per_mm2_V1('4B'))

    # print(y19.get_centre('V1'))
    # print(y19.get_centre('V2'))
    # print(y19.get_centre('V4'))
    # print(y19.get_centre('MT'))

    # print(y19.midthickness_points.shape)
    # print(y19.midthickness_triangles.shape)
    # print(y19.very_inflated_points.shape)
    # print(y19.areas)
    # print(y19.label_inds.shape)
    # print(y19.labels.shape)

    # sources, targets, fractions = _read_fraction_labelled_neurons_extrinsic()
    # print('{} to {}: {}'.format(sources[0], targets[0], fractions[0]))
    #
    # sources, targets, percents = _read_supragranular_layers_percent()
    # print('{} to {}: {}'.format(sources[0], targets[0], percents[0]))
    # print(np.max(percents))

    # print(_get_synapses_per_neuron('p2/3'))
    # print(_get_synapses_per_neuron_V1())
    # print(_get_synapses_per_neuron_cat_V1())

    # _plot_synapses_per_layer_per_mm_2()
    # print(_get_synapses_per_layer_cat_V1('6'))
    # print(_synapses_per_neuron_V1())
    # print(_synapses_per_neuron_cat_V1())

    # _get_synapses_per_layer_V1('4')
    # _get_synapses_per_layer_V2('6')

    # print(synapses_per_neuron('V2', '4', '2/3'))
    # print(synapses_per_neuron('V2', '2/3', '5'))
    # print(synapses_per_neuron('MT', 'extrinsic', '6'))

    # c = CoCoMac()
    # # print('FV91-{}'.format(c._map_M14_to_FV91('V1')))
    # # print('FV91-{}'.format(c._map_M14_to_FV91('MST')))
    # layers = c.data['FV91-{}'.format(c._map_M14_to_FV91('V1'))]['FV91-{}'.format(c._map_M14_to_FV91('MT'))]
    # print(layers)
    # # print(c._guess_missing_layers('V1', 'MSTd', layers))
    # # print(c.get_connection_details('V1', 'V4'))
    #
    # # c._print_fraction_asymmetric('6')
    # # print(c._map_axon_termination_layers_to_cell_layers([True, False, False, False, False, True]))

    # m = Markov()
    # print(m.get_SLN('V2', 'TEpd'))
    # print(m.get_SLN('V3', 'TEpd'))

    # calculate_mean_ascending_SLN()

    # calculate_flne_vs_distance()

    # m = Markov()
    # target = 'V3'
    # print(m.is_injection_site(target))
    # print(m.get_sources(target))
    # sources = m.get_sources_with_fallback(target)
    # print(sources)
    # for source in sources:
    #     FLNe = m.get_FLNe(source, target)
    #     SLN = m.get_SLN(source, target)
    #     print('{}->{} FLNe: {} %SLN: {}'.format(source, target, FLNe, SLN))
