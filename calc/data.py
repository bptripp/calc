import os
import inspect
import nibabel
import numpy as np
import xml.etree.ElementTree as ET
import csv
import json
from scipy.optimize import curve_fit

class Data:
    def __init__(self):
        iac = InterAreaConnections()
        self.FLNe = iac.get_interpolated_FLNe()
        self.SLN = iac.get_interpolated_SLN()
        self.connections = iac.get_connectivity_grid()

        histogram = np.array(FS09_synapses_per_connection)
        self.synapses_per_connection = np.dot(histogram[:,0], histogram[:,1]) / np.sum(histogram[:,1])

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
            assume convolutional units are similar to inhibitory neurons based on Parisien et al. (2008).
        """
        #TODO: account for lack of neurons in L1
        #TODO: account for varied density across layers
        mm2 = S18_surface_area[area]

        if area == 'V1':
            neurons_per_mm2 = _get_neurons_per_mm2_V1(layer)
        else:
            total_neurons_per_mm2 = S18_density[area][1]
            layer_index = ['1', '2/3', '4', '5', '6'].index(layer)
            fraction = S18_thickness[area][layer_index] / S18_thickness[area][-1]
            neurons_per_mm2 = fraction * total_neurons_per_mm2

        # We multiply by 0.75 to match fraction excitatory cells; see Hendry et al. (1987) J Neurosci
        return int(0.75 * mm2 * neurons_per_mm2)

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
        spn = synapses_per_neuron(area, source_layer, target_layer)
        return spn / self.synapses_per_connection

    def get_extrinsic_inputs(self, area, target_layer):
        """
        :param area: A visual area
        :param target_layer: A cortical layer
        :return: Number of inputs per neuron from other cortical areas
        """
        #TODO: account for repeated synapses
        spn = synapses_per_neuron(area, 'extrinsic', target_layer)
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

    def get_interpolated_FLNe(self):
        """
        :return: FLNe grid for FV91 visual areas. For areas injected by Markov et al.,
            FLNe is estimated according to fraction overlap between M132 areas (used by
            Markov et al.) and FV91 areas (used here). For other areas, we interpolate
            based on inter-area distances through white matter. The distances are
            provided by Schmidt et al.
        """
        remapped_FLNe = self.get_remapped_FLNe() #data from Markov, mapped onto FV91 areas

        # calculate curve fit to estimate connections to areas not injected by Markov et al.
        logFLNe = np.log10(remapped_FLNe).flatten()
        dist = np.array(S18_distance).flatten()
        ind = np.where(np.logical_and(dist > 0, np.isfinite(logFLNe)))
        logFLNe = logFLNe[ind]
        dist = dist[ind]
        coeffs = np.polyfit(dist, logFLNe, 1)

        grid = self.get_connectivity_grid()

        interpolated_FLNe = np.exp(coeffs[1] + coeffs[0] * np.array(S18_distance))
        # interpolated_FLNe = np.multiply(interpolated_FLNe, grid)

        remapped_rows = np.isfinite(np.sum(remapped_FLNe, axis=1))
        remapped_totals = np.zeros(remapped_rows.shape)

        for i in range(len(remapped_rows)):
            if remapped_rows[i]:
                interpolated_FLNe[i,:] = remapped_FLNe[i,:]
                remapped_totals[i] = np.sum(remapped_FLNe[i,:])

        interpolated_FLNe = np.multiply(interpolated_FLNe, grid)

        # for rows from Markov et al., recover sum of FLNe from before removing dubious connections
        for i in range(len(remapped_rows)):
            if remapped_rows[i]:
                interpolated_FLNe[i,:] = interpolated_FLNe[i,:] / remapped_totals[i]

        return interpolated_FLNe

    def get_remapped_FLNe(self):
        """
        :return: FLNe grid for FV91 areas, including only areas injected by Markov et al. They report
            FLNe for the M132 parcellation, but here we attempt to map onto FV91 according to degree
            overlap from CoCoMac.
        """
        n = len(self.areas)
        grid = np.zeros((n, n))
        for i in range(n):
            sites = self.get_M132_injection_sites(self.areas[i])
            mapped = []
            for j in range(len(sites)):
                M132_FLNe = self.get_M132_FLNe(sites[j])
                mapped.append(self.map_FLNe_M132_to_FV91(M132_FLNe))
            grid[i,:] = np.mean(mapped, axis=0)
            grid[i,i] = 0
        return grid

    def map_FLNe_M132_to_FV91(self, M132_FLNe):
        """
        :param M132_FLNe: result of get_M132_FLNe() for a certain target
        :return: list of FLNe for each area in areas_FV91
        """
        result = np.zeros(len(areas_FV91))
        for source_M132 in M132_FLNe.keys():
            FLNe = M132_FLNe[source_M132]
            fractions = self._get_overlap_fractions(source_M132)
            result += FLNe * fractions
        return result

    def get_M132_FLNe(self, target):
        """
        :param target: A M132 area
        :return: a dictionary with M132 connected areas as keys and FLNe as values
        """
        result = dict()
        for source in self.markov.get_sources(target):
            result[source] = self.markov.get_FLNe(source, target)
        return result

    def get_interpolated_SLN(self):
        """
        :return: SLN grid for FV91 visual areas. For areas injected by Markov et al.,
            SLN is estimated according to fraction overlap between M132 areas (used by
            Markov et al.) and FV91 areas (used here). For other areas, we estimate
            (following Schmidt et al.) based on ratio of neuron densities in each area.
        """

        remapped_SLN = self.get_remapped_SLN()

        # find curve fit to extrapolate to non-injected areas
        density_log_ratios = []
        SLNs = []
        n = len(areas_FV91)
        density_ratios = self._get_density_ratio_grid(interpolate=False)
        for i in range(n):
            for j in range(n):
                if np.isfinite(density_ratios[i,j]) and remapped_SLN[i,j] > 0:
                    log_ratio = np.log(density_ratios[i,j])
                    density_log_ratios.append(log_ratio)
                    SLNs.append(remapped_SLN[i, j])
        popt, pcov = curve_fit(sigmoid, density_log_ratios, np.array(SLNs)/100, p0=[0, -1])

        grid = self.get_connectivity_grid()
        density_ratios = self._get_density_ratio_grid(interpolate=True)
        interpolated_SLN = 100*sigmoid(np.log(density_ratios), popt[0], popt[1])

        # plt.scatter(np.log(density_ratios.flatten()), interpolated_SLN.flatten())
        # plt.scatter(density_log_ratios, SLNs)
        # x = np.linspace(-2, 2, 20)
        # plt.plot(x, 100*sigmoid(x, popt[0], popt[1]))
        # plt.show()

        for i in range(n):
            if self.get_M132_injection_sites(self.areas[i]):
                interpolated_SLN[i,:] = remapped_SLN[i,:]

        result = np.multiply(interpolated_SLN, grid)
        result[np.where(result == 0)] = np.nan
        return result

    def _get_density_ratio_grid(self, interpolate=False):
        n = len(areas_FV91)

        result = np.nan * np.zeros((n,n))
        for i in range(n):
            density_i = self._get_density(areas_FV91[i], interpolate=interpolate)
            for j in range(n):
                density_j = self._get_density(areas_FV91[j], interpolate=interpolate)
                if density_i and density_j:
                    result[i,j] = density_i / density_j

        return result

    def _get_density(self, area_FV91, interpolate=False):
        """
        :param area_FV91: a FV91 area
        :param interpolate (False): if True, fit missing results as a function of distance
            from V1 (constant for >30mm; linear fit for closer areas)
        :return: neurons/mm^3 if available, otherwise None or an estimate if interpolate=True
        """
        result = None

        if area_FV91 in S18_density.keys():
            result = S18_density[area_FV91][1]

        if interpolate and result is None:
            distance_from_V1 = S18_distance[0][areas_FV91.index(area_FV91)]
            if distance_from_V1 > 30:
                result = 39727.5
            else:
                result = 152728.28141914 - 3796.74396312*distance_from_V1

        return result

    def get_remapped_SLN(self):
        """
        :return: SLN grid for FV91 areas, including only areas injected by Markov et al. They report
            SLN for the M132 parcellation, but here we attempt to map onto FV91 according to degree
            overlap from CoCoMac.
        """
        n = len(self.areas)
        grid = np.zeros((n, n))
        for i in range(n):
            sites = self.get_M132_injection_sites(self.areas[i])
            mapped = []
            for j in range(len(sites)):
                M132_SLN = self.get_M132_SLN(sites[j])
                mapped.append(self.map_SLN_M132_to_FV91(sites[j], M132_SLN))
            grid[i,:] = np.mean(mapped, axis=0)
            grid[i,i] = 0
        return grid

    def map_SLN_M132_to_FV91(self, target_M132, M132_SLN):
        """
        This implements the equation on pg 1416 of Schmidt et al. (2018).

        :param M132_SLN: result of get_M132_SLN() for a certain target area
        :param M132_target: the target area
        :return: list of SLN for each area in areas_FV91
        """
        numerator = np.zeros(len(areas_FV91))
        denominator = np.zeros(len(areas_FV91))
        for source_M132 in M132_SLN.keys():
            FLNe = self.markov.get_FLNe(source_M132, target_M132)
            SLN = M132_SLN[source_M132]
            fractions = self._get_overlap_fractions(source_M132)
            numerator += FLNe * SLN * fractions
            denominator += FLNe * fractions
        return np.divide(numerator, denominator)

    def _get_overlap_fractions(self, area_M132):
        """
        :param area_M132: A M132 area
        :return: fraction overlap between given M132 area and each FV91 area (in order of areas_FV91)
        """
        fractions = np.zeros(len(areas_FV91))
        percents = self.cocomac.get_M132_to_FV91(area_M132)
        if percents:
            for area_FV91 in percents.keys():
                if area_FV91 in areas_FV91:
                    ind = areas_FV91.index(area_FV91)
                    fractions[ind] += percents[area_FV91] / 100
        return fractions

    def get_M132_injection_sites(self, FV91_area):
        """
        :param FV91_area: An area in the FV91 parcellation
        :return: M132 areas of injection sites in Markov et al. that fall in the FV91_area
            (from Schmidt et al.)
        """
        result = []
        for key in S18_injection_site_mappings.keys():
            if S18_injection_site_mappings[key] == FV91_area:
                result.append(key)
        return result

    def get_M132_SLN(self, target):
        """
        :param target: A M132 area
        :return: a dictionary with M132 connected areas as keys and %SLN as values
        """
        result = dict()
        for source in self.markov.get_sources(target):
            result[source] = self.markov.get_SLN(source, target)
        return result


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

# (intrinsic, extrinsic) for (2/3E, 2/3I, 4E, 4I, 5E, 5I, 6E, 6I)
#TODO: use Binzegger's numbers for V1 but scale total layer-wise by Schmidt's layer-wise in-degree
#TODO: Schmidt's estimate based on layer-specific synapse densitites and layer thicknesses
#TODO: but is this table by layer in which synapse occurs or layer of cell body? L1 missing so must be latter

S18_in_degree = {
    'V1': [3550.00,1246.00,2885.00,1246.00,1975.00,1246.00,2860.00,1246.00,4100.00,1246.00,1632.00,1246.00,2008.00,1246.00,1644.00,1246.00],
    'V2': [3608.00,1848.00,3853.00,1848.00,3413.00,1848.00,4819.00,1848.00,5669.00,1848.00,3124.00,1848.00,4596.00,1848.00,3511.00,1848.00],
    'VP': [4345.00,1756.00,4345.00,1756.00,3455.00,1756.00,4233.00,1756.00,6012.00,1756.00,2598.00,1756.00,3383.00,1756.00,2605.00,1756.00],
    'V3': [4227.00,1810.00,4270.00,1810.00,3833.00,1810.00,4664.00,1810.00,6341.00,1810.00,2576.00,1810.00,3618.00,1810.00,2558.00,1810.00],
    'V3A': [6086.00,2703.00,6347.00,2703.00,7114.00,2703.00,8001.00,2703.00,7881.00,2703.00,3714.00,2703.00,4786.00,2703.00,3587.00,2703.00],
    'MT': [5530.00,2510.00,5685.00,2510.00,6383.00,2510.00,6841.00,2510.00,7557.00,2510.00,3372.00,2510.00,4537.00,2510.00,3326.00,2510.00],
    'V4t': [5700.00,2293.00,6234.00,2293.00,5856.00,2293.00,6867.00,2293.00,7815.00,2293.00,3843.00,2293.00,4952.00,2293.00,3795.00,2293.00],
    'V4': [4749.00,2337.00,5074.00,2337.00,5481.00,2337.00,5861.00,2337.00,7051.00,2337.00,3272.00,2337.00,4769.00,2337.00,3453.00,2337.00],
    'VOT': [5065.00,2409.00,5346.00,2409.00,7426.00,2409.00,9952.00,2409.00,5375.00,2409.00,2786.00,2409.00,3713.00,2409.00,2462.00,2409.00],
    'MSTd': [7356.00,3181.00,7219.00,3181.00,8903.00,3181.00,9986.00,3181.00,8606.00,3181.00,3938.00,3181.00,4714.00,3181.00,3764.00,3181.00],
    'PIP': [6913.00,3327.00,7216.00,3327.00,8900.00,3327.00,10165.00,3327.00,8286.00,3327.00,4069.00,3327.00,4971.00,3327.00,3859.00,3327.00],
    'PO': [7482.00,3226.00,7432.00,3226.00,8083.00,3226.00,8943.00,3226.00,9001.00,3226.00,4167.00,3226.00,4879.00,3226.00,4033.00,3226.00],
    'DP': [7751.00,3328.00,7793.00,3328.00,9097.00,3328.00,9133.00,3328.00,9596.00,3328.00,4477.00,3328.00,5249.00,3328.00,4385.00,3328.00],
    'MIP': [8244.00,3474.00,7919.00,3474.00,8191.00,3474.00,8911.00,3474.00,10903.00,3474.00,4303.00,3474.00,4547.00,3474.00,4105.00,3474.00],
    'MDP': [6349.00,5186.00,6702.00,5186.00,3587.00,5186.00,7457.00,5186.00,6246.00,5186.00,3493.00,5186.00,3271.00,5186.00,3086.00,5186.00],
    'VIP': [6602.00,3378.00,6777.00,3378.00,7163.00,3378.00,8095.00,3378.00,9069.00,3378.00,3939.00,3378.00,5653.00,3378.00,4078.00,3378.00],
    'LIP': [7331.00,3311.00,7438.00,3311.00,8690.00,3311.00,8926.00,3311.00,9781.00,3311.00,4362.00,3311.00,4623.00,3311.00,3910.00,3311.00],
    'PITv': [6108.00,2441.00,5906.00,2441.00,5602.00,2441.00,7010.00,2441.00,7243.00,2441.00,3231.00,2441.00,3892.00,2441.00,3136.00,2441.00],
    'PITd': [5820.00,2471.00,5721.00,2471.00,6000.00,2471.00,7663.00,2471.00,6760.00,2471.00,3105.00,2471.00,3818.00,2471.00,2957.00,2471.00],
    'MSTl': [7491.00,3094.00,7482.00,3094.00,8566.00,3094.00,9595.00,3094.00,8935.00,3094.00,4122.00,3094.00,5013.00,3094.00,3917.00,3094.00],
    'CITv': [8696.00,3844.00,8567.00,3844.00,12863.00,3844.00,13354.00,3844.00,9926.00,3844.00,4627.00,3844.00,5434.00,3844.00,4387.00,3844.00],
    'CITd': [7641.00,3708.00,8066.00,3708.00,17442.00,3708.00,20485.00,3708.00,8023.00,3708.00,4204.00,3708.00,5357.00,3708.00,3714.00,3708.00],
    'FEF': [7499.00,3597.00,7936.00,3597.00,9253.00,3597.00,9708.00,3597.00,8286.00,3597.00,4003.00,3597.00,4634.00,3597.00,3802.00,3597.00],
    'TF': [7497.00,3805.00,7692.00,3805.00,8692.00,3805.00,10184.00,3805.00,8790.00,3805.00,4268.00,3805.00,5135.00,3805.00,4027.00,3805.00],
    'AITv': [8947.00,3786.00,8716.00,3786.00,12235.00,3786.00,12248.00,3786.00,10346.00,3786.00,4735.00,3786.00,5498.00,3786.00,4539.00,3786.00],
    'FST': [9905.00,4614.00,10189.00,4614.00,14721.00,4614.00,15183.00,4614.00,11516.00,4614.00,5671.00,4614.00,6641.00,4614.00,5428.00,4614.00],
    '7a': [9280.00,4361.00,9450.00,4361.00,14158.00,4361.00,12136.00,4361.00,11391.00,4361.00,5446.00,4361.00,6207.00,4361.00,5206.00,4361.00],
    'STPp': [8147.00,4246.00,8771.00,4246.00,14959.00,4246.00,15201.00,4246.00,9707.00,4246.00,5026.00,4246.00,5931.00,4246.00,4669.00,4246.00],
    'STPa': [8283.00,4032.00,8546.00,4032.00,17072.00,4032.00,18775.00,4032.00,9054.00,4032.00,4548.00,4032.00,5531.00,4032.00,4151.00,4032.00],
    '46': [8562.00,4309.00,9443.00,4309.00,12826.00,4309.00,11556.00,4309.00,10709.00,4309.00,5580.00,4309.00,6265.00,4309.00,5267.00,4309.00],
    'AITd': [9256.00,3784.00,8883.00,3784.00,11106.00,3784.00,10468.00,3784.00,10878.00,3784.00,4865.00,3784.00,5540.00,3784.00,4731.00,3784.00],
    'TH': [9229.00,5491.00,9829.00,5491.00,0,0,0,0,9468.00,5491.00,4774.00,5491.00,6566.00,5491.00,5629.00,5491.00]
}

S18_thickness = {
    'V1': [0.09,0.37,0.46,0.17,0.16,1.24],
    'V2': [0.12,0.6,0.24,0.25,0.25,1.46],
    'VP': [0.18,0.63,0.32,0.21,0.25,1.59],
    'V3': [0.23,0.7,0.31,0.16,0.19,1.59],
    'V3A': [0.2,0.71,0.24,0.23,0.28,1.66],
    'MT': [0.2,0.95,0.26,0.26,0.29,1.96],
    'V4t': [0.22,0.8,0.29,0.26,0.31,1.88],
    'V4': [0.18,1,0.24,0.24,0.24,1.89],
    'VOT': [0.23,0.81,0.28,0.27,0.32,1.9],
    'MSTd': [0.26,0.92,0.24,0.3,0.36,2.07],
    'PIP': [0.26,0.92,0.24,0.3,0.36,2.07],
    'PO': [0.26,0.92,0.24,0.3,0.36,2.07],
    'DP': [0.26,0.91,0.23,0.3,0.36,2.06],
    'MIP': [0.2,0.85,0.17,0.16,0.7,2.07],
    'MDP': [0.26,0.92,0.24,0.3,0.36,2.07],
    'VIP': [0.25,1.17,0.28,0.21,0.16,2.07],
    'LIP': [0.25,1,0.24,0.24,0.57,2.3],
    'PITv': [0.23,0.81,0.28,0.27,0.32,1.9],
    'PITd': [0.23,0.81,0.28,0.27,0.32,1.9],
    'MSTl': [0.26,0.92,0.24,0.3,0.36,2.07],
    'CITv': [0.29,1.02,0.19,0.33,0.4,2.23],
    'CITd': [0.29,1.02,0.19,0.33,0.4,2.23],
    'FEF': [0.22,0.92,0.35,0.37,0.35,2.21],
    'TF': [0.23,0.66,0.21,0.24,0.28,1.62],
    'AITv': [0.34,1.2,0.23,0.39,0.47,2.63],
    'FST': [0.51,0.9,0.18,0.3,0.36,2.25],
    '7a': [0.35,1.24,0.21,0.41,0.48,2.68],
    'STPp': [0.29,1.03,0.18,0.34,0.4,2.25],
    'STPa': [0.29,1.03,0.18,0.34,0.4,2.25],
    '46': [0.22,0.82,0.18,0.28,0.36,1.86],
    'AITd': [0.34,1.2,0.23,0.39,0.47,2.63],
    'TH': [0.28,0.65,0.12,0.57,0.26,1.87]
}

S18_surface_area = {
    'V1': 1484.63, #TODO: this doesn't match FV91
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

# Mappings from Markov et al. injections sites in M132 areas to FV91 areas.
# Some mappings are many-to-one.
S18_injection_site_mappings = {
    'V1': 'V1',
    'V2': 'V2',
    'V4': 'V4',
    'STPr': 'STPa',
    'TEO': 'V4',
    'STPi': 'STPp',
    '7A': '7a',
    'STPc': 'STPp',
    'DP': 'DP',
    '9/46d': 'FEF',
    '8l': 'FEF',
    '8m': 'FEF',
    'MT': 'MSTd',
    '46d': '46',
    '9/46v': '46',
    'TEpd': 'CITv',
    'PBr': 'STPp'
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

# Structural type, neurons/mm^2, and total thickness.
# From Hilgetag et al. (2016) with area mappings suggested by Schmidt et al. (2018)
# Hilgetag Claus C., et al. "The primate connectome in context: principles of connections
# of the cortical visual system." NeuroImage 134 (2016): 685-702,
# Areas marked with "inferred" are not in the dataset, so we use the mean of other areas
# at the same hierarchical level, following Schmidt et al.
# Areas marked "manual" were manually assigned type 5 by Schmidt et al.
S18_density = {
    'V1': [8, 161365,1.24],
    'V2': [7,97619,1.46],
    'V3': [7,97619,None], # inferred
    'VP': [7,97619,None], # inferred
    'V4': [6,71237,1.89],
    'MT': [6,65992,1.96],
    'VOT': [6,63271,2.13],
    'PITd': [6,63271,2.13],
    'PITv': [6,63271,2.13],
    'V3A': [6,61382,1.66],
    'V4t': [6,64737,None], # inferred
    'MIP': [5,47137,None], # manual inferred
    'MDP': [5,47137,None], # manual inferred
    'LIP': [5,(53706+45237)/2,2.3],
    'DP': [5,48015,2.06],
    'TF': [5,46084,1.62],
    'FEF': [5,44978,2.21],
    'CIT': [5,47137,None], # inferred
    'MSTd': [5,47137,None], # inferred
    'MSTl': [5,47137,None], # inferred
    'PIP': [5,47137,None], # inferred
    'PITd': [5,47137,None], # inferred
    'PITv': [5,47137,None], # inferred
    'PO': [5,47137,None], # inferred
    'VIP': [5,47137,None], # inferred
    'AITd': [4,38840,2.63],
    'AITv': [4,38840,2.63],
    'CITd': [4,38840,2.63],
    'CITv': [4,38840,2.63],
    '46': [4,38027,1.86],
    '7a': [4,36230,2.68],
    'FST': [4,38269,None], # inferred
    'STPa': [4,38269,None], # inferred
    'STPp': [4,38269,None], # inferred
    'TH': [2,33196,1.87],
}

"""
Account for spine density vs hierarchy as in: 
Chaudhuri, Rishidev, et al. "A large-scale circuit mechanism for hierarchical dynamical processing 
in the primate cortex." Neuron 88.2 (2015): 419-431.

Omit deep neurons in early areas that project to SC: 
Fries, Wolfgang. "Cortical projections to the superior colliculus in the macaque monkey: a retrograde 
study using horseradish peroxidase." Journal of Comparative Neurology 230.1 (1984): 55-76.
Hübener, Mark, Cornelius Schwarz, and Jürgen Bolz. "Morphological types of projection neurons in layer 
5 of cat visual cortex." Journal of comparative neurology 301.4 (1990): 655-674.

Also omit callosal neurons.
J. C. Houzel, M. L. Carvalho, and R. Lent, “Interhemispheric connections between primary visual 
areas: Beyond the midline rule,” Brazilian J. Med. Biol. Res., vol. 35, no. 12, pp. 1441–1453, 2002.

Also account for non-projecting pyramidal cells: 
Gilbert, Charles D., and TORSTEN N. Wiesel. "Clustered intrinsic connections in cat visual cortex." 
Journal of Neuroscience 3.5 (1983): 1116-1133.

"""

"""
Where possible, connection strengths and origin layers are taken from:

Markov, N. T., Ercsey-Ravasz, M. M., Ribeiro Gomes, A. R., Lamy, C., Magrou, L., Vezoli, 
J., ... & Sallet, J. (2012). A weighted and directed interareal connectivity matrix for macaque 
cerebral cortex. Cerebral cortex, 24(1), 17-36.

Additional connections and termination layers are taken from CoCoMac 2.0:
 
Bakker, R., Wachtler, T., & Diesmann, M. (2012). CoCoMac 2.0 and the future of tract-tracing 
databases. Frontiers in neuroinformatics, 6.

A major source for information on inter-laminar connections (within a single area) is: 

T. Binzegger, R. J. Douglas, and K. A. C. Martin, “A quantitative map of the circuit of cat 
primary visual cortex,” J. Neurosci., vol. 24, no. 39, pp. 8441–8453, 2004.

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
    'extrinsic': ['X/Y', 'as']
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

        self._fit()

    def _fit(self):
        centre_V1 = yerkes19.get_centre('V1')
        distances = []
        spine_counts = []
        for key in self.layer_3_basal_spine_count.keys():
            if key in yerkes19.areas:
                areas = [key]
            else:
                areas = self.yerkes_mappings[key]

            centres = []
            for area in areas:
                centres.append(yerkes19.get_centre(area))
            distance = np.linalg.norm(centre_V1 - np.mean(centres, axis=0))
            distances.append(distance)
            spine_counts.append(self.layer_3_basal_spine_count[key])

        p = np.polyfit(distances, spine_counts, 1)
        self.intercept = p[1]
        self.slope = p[0]

        # import matplotlib.pyplot as plt
        # plt.plot(distances, spine_counts, '.')
        # plt.plot([0, 50], p[1] + [0, p[0]*50], 'k')
        # plt.show()

    def get_spine_count(self, area):
        if area in self.layer_3_basal_spine_count.keys():
            return self.layer_3_basal_spine_count[area]
        else:
            for key in self.yerkes_mappings.keys():
                if area in self.yerkes_mappings[key]:
                    return self.layer_3_basal_spine_count[key]

            if area in yerkes19.areas:
                centre_V1 = yerkes19.get_centre('V1')
                centre_area = yerkes19.get_centre(area)
                distance = np.linalg.norm(centre_V1 - centre_area)
                return self.intercept + self.slope*distance
            else:
                return None


def plot_spine_count_estimates():
    e07 = E07()
    centre_V1 = yerkes19.get_centre('V1')
    distances = []
    spine_counts = []
    for area in yerkes19.areas:
        print(area)
        centre = yerkes19.get_centre(area)
        distances.append(np.linalg.norm(centre_V1 - centre))
        spine_counts.append(e07.get_spine_count(area))

    import matplotlib.pyplot as plt
    plt.plot(distances, spine_counts, '.')
    plt.show()


def synapses_per_neuron(area, source_layer, target_layer):
    """
    Mean inbound connections per neuron. Only excitatory cells are considered, based on the
    rationale in:

    Parisien, C., Anderson, C. H., & Eliasmith, C. (2008). Solving the problem of negative
    synaptic weights in cortical models. Neural computation, 20(6), 1473-1494.
    Tripp, B., & Eliasmith, C. (2016). Function approximation in inhibitory networks.
    Neural Networks, 77, 95-106.

    Inputs to excitatory cells are averaged over excitatory cell types, weighted by numbers
    of each cell type.

    The numbers are based on potential (geometric) connections in cat V1, from:

    T. Binzegger, R. J. Douglas, and K. A. C. Martin, “A quantitative map of the circuit of cat
    primary visual cortex,” J. Neurosci., vol. 24, no. 39, pp. 8441–8453, 2004.

    Regarding similarity of inter-laminar structure between cat and macaque V1, see:

    E. M. Callaway, “Local Circuits in Primary Visual Cortex of the Macaque Monkey,”
    Annu. Rev. Neurosci., vol. 21, no. 1, pp. 47–74, 1998.

    Regarding support for random contact (but not functional connectivity) based on geometry see:

    N. Kalisman, G. Silberberg, and H. Markram, “The neocortical microcircuit as a tabula
    rasa,” Proc. Natl. Acad. Sci., vol. 102, no. 3, pp. 880–885, 2005.

    :param area: cortical area (e.g. 'V1', 'V2')
    :param source_layer: one of '1', '2/3', '4', '5', '6' or 'extrinsic'
    :param target_layer: one of '1', '2/3', '4', '5', '6'
    :return:
    """
    n_source_types = 16
    n_target_types = 19

    # find table of synapses between types summed across all layers
    totals = np.zeros((n_target_types, n_source_types))
    layers = ['1', '2/3', '4', '5', '6']
    if area == 'V1':
        for layer in layers:
            totals = totals + _get_synapses_per_layer_V1(layer)
    else:
        for layer in layers:
            totals = totals + _get_synapses_per_layer_V2(layer)

    # sum over sources and weighted average over targets ...
    source_types = BDM04_excitatory_types[source_layer] # cell types in source layer (regardless of where synapses are)
    target_types = BDM04_excitatory_types[target_layer]

    total_inputs = np.zeros(n_target_types)
    for i in range(n_source_types):
        if BDM04_sources[i] in source_types:
            total_inputs = total_inputs + totals[:,i]

    numerator = 0.
    denominator = 0.
    for i in range(n_target_types):
        if BDM04_targets[i] in target_types:
            n = BDM04_N_per_type[BDM04_targets[i]]
            numerator = numerator + n * total_inputs[i]
            denominator = denominator + n

    result = numerator / denominator

    # # for areas other than V1, scale with spine density
    # ratio = e07.get_spine_count(area) / e07.get_spine_count('V1')
    # return ratio * result

    # For areas other than V1, scale by in-degree of target layer estimated by Schmidt et al.
    col = 2 * ['2/3', '4', '5', '6'].index(target_layer) # column of S18_in_degree for excitatory cells
    ratio = S18_in_degree[area][col] / S18_in_degree['V1'][col]
    return ratio * result


def _synapses_per_neuron_V1():
    """
    :return: Mean # of synapses per neuron across all monkey V1 (from OC82 data)
    """
    total_neurons_per_mm2 = 0
    for value in OC82_n_neurons_per_mm2_V1.values():
        total_neurons_per_mm2 = total_neurons_per_mm2 + value

    total_synapses_per_mm2 = 0
    for value in OC82_n_synapses_per_mm2_V1.values():
        total_synapses_per_mm2 = total_synapses_per_mm2 + value

    return total_synapses_per_mm2 / total_neurons_per_mm2


def _synapses_per_neuron_cat_V1():
    """
    :return: Mean # of synapses per neuron across all cat V1 (from BDM04 data)
    """
    total_synapses_table = np.zeros((19,16))
    for layer in ['1', '2/3', '4', '5', '6']:
        total_synapses_table = total_synapses_table + _get_synapses_per_layer_cat_V1(layer)

    total_neurons = 0
    total_synapses = 0
    for i in range(19):
        type = BDM04_targets[i]
        n_per_hemiphere = BDM04_N_per_type[type]
        total_neurons = total_neurons + n_per_hemiphere
        total_synapses = total_synapses + n_per_hemiphere * np.sum(total_synapses_table[i])

    return total_synapses / total_neurons


def _get_synapses_per_layer_V1(layer):
    """
    :param layer: one of 1, 2/3, 4, 5, 6
    :return: Table of synapses per layer, from BDM04 cat data, but scaled according
        to lower mean synapses per neuron in monkey
    """
    monkey_cat_ratio = _synapses_per_neuron_V1() / _synapses_per_neuron_cat_V1()
    return monkey_cat_ratio * _get_synapses_per_layer_cat_V1(layer)


def _get_synapses_per_layer_V2(layer, rescale=True):
    """
    :param layer: one of 1, 2/3, 4, 5, 6
    :param rescale: if True (the default), rescale cat V1 data by layer thickness. It isn't clear
        that this is the right thing to do. For example, it seems doubtful that the relatively
        sparse extrinsic inputs to L4 are more numerous with thicker L4, as they take up little
        of the volume. However, synapse density per mm^3 is remarkably uniform across areas and
        species, so density per mm^2 increases with thickness.
        DeFelipe, J., Alonso-Nanclares, L., & Arellano, J. I. (2002). Microstructure of the
        neocortex: comparative aspects. Journal of neurocytology, 31(3-5), 299-316.
    :return: Table of estimated synapses per layer, based on data from cat V1. More direct
        information would be useful here. The justification is that cat V1 is much like primate
        extrastriate areas in terms of neuron density per mm^2 (whereas it is unlike primate V1):
        Srinivasan, S., Carlo, C. N., & Stevens, C. F. (2015). Predicting visual acuity from the
        structure of visual cortex. PNAS, 112(25), 7815-7820.
        Carlo, C. N., & Stevens, C. F. (2013). Structural uniformity of neocortex, revisited.
        PNAS, 110(4), 1488-1493.

    """
    if rescale:
        if layer == '2/3':
            thickness_V1 = BYK14_thickness_V1['2'] + BYK14_thickness_V1['3A'] + BYK14_thickness_V1['3B']
        else:
            thickness_V1 = 0
            for key in BYK14_thickness_V1.keys():
                if key.startswith(layer):
                    thickness_V1 = thickness_V1 + BYK14_thickness_V1[key]

        thickness_V2 = BYK14_thickness_V2[layer]

        return thickness_V2 / thickness_V1 * _get_synapses_per_layer_cat_V1(layer)
    else:
        return _get_synapses_per_layer_cat_V1(layer)


def _get_synapses_per_layer_cat_V1(layer):
    with open(data_folder() + '/BDM04-Supplementary.txt') as file:
        found_layer = False
        table = []
        while True:
            line = file.readline()

            if not line:
                break
            if found_layer and len(line.strip()) == 0:
                break

            if not line.startswith('#'): # comment
                if found_layer:
                    n_cols = 16
                    items = line.split()
                    row = np.zeros(n_cols)
                    assert len(items) == n_cols+1 or len(items) == 1 # expected # columns or empty
                    for i in range(1, len(items)):  # skip row header
                        row[i-1] = float(items[i].replace('-', '0'))
                    table.append(row)

                if line.startswith('L{}'.format(layer)):
                    found_layer = True

        assert len(table) == 19 # expected # of rows
        return np.array(table)


def _get_synapses_per_neuron_per_type_cat_V1(type):
    total = 0
    total = total + BDM04_synapses_per_neuron_L1.get(type, 0)
    total = total + BDM04_synapses_per_neuron_L23.get(type, 0)
    total = total + BDM04_synapses_per_neuron_L4.get(type, 0)
    total = total + BDM04_synapses_per_neuron_L5.get(type, 0)
    total = total + BDM04_synapses_per_neuron_L6.get(type, 0)
    return total


def _get_n_neurons_per_mm_2_cat_V1(type):
    return BDM04_N_per_type[type] / BDM04_total_area_hemisphere


def _get_synapses_per_mm_2_V1(layer):
    #TODO: test
    total = 0
    for key in OC82_n_synapses_per_mm2_V1.keys():
        if key.startswith(layer):
            total = total + OC82_n_synapses_per_mm2_V1[key]
    return total

def _get_synapses_per_neuron_cat_V1():
    neurons = 0
    synapses = 0
    for key in BDM04_N_per_type.keys():
        nt = BDM04_N_per_type[key] / BDM04_total_area_hemisphere
        neurons = neurons + nt
        synapses = synapses + nt * _get_synapses_per_neuron_per_type_cat_V1(key)

    # print('cat neurons: {} synapses: {}'.format(neurons, synapses))
    return synapses / neurons


def _get_synapses_per_neuron_V1():
    #TODO: difference seems consistent with Colonnier & O'Kusky (1981) RCB
    #TODO: see Carlo & Stevens (2012)
    neurons = 0
    for value in OC82_n_neurons_per_mm2_V1.values():
        neurons = neurons + value

    synapses = 0
    for value in OC82_n_synapses_per_mm2_V1.values():
        synapses = synapses + value

    # print('monkey neurons: {} synapses: {}'.format(neurons, synapses))
    return synapses / neurons


def _plot_synapses_per_layer_per_mm_2():
    layers = ['1', '2/3', '4', '5', '6']

    monkey_counts = []
    for layer in layers:
        count = 0
        for key in OC82_n_synapses_per_mm2_V1.keys():
            if key.startswith(layer): 
                count = count + OC82_n_synapses_per_mm2_V1[key]
        monkey_counts.append(count)

    cat_counts = []
    for layer in layers: 
        if layer == '1': 
            synapses_per_neuron = BDM04_synapses_per_neuron_L1
        elif layer == '23': 
            synapses_per_neuron = BDM04_synapses_per_neuron_L23
        elif layer == '4':
            synapses_per_neuron = BDM04_synapses_per_neuron_L4
        elif layer == '5': 
            synapses_per_neuron = BDM04_synapses_per_neuron_L5
        else:
            synapses_per_neuron = BDM04_synapses_per_neuron_L6

        count = 0
        for key in synapses_per_neuron.keys():
            neurons = BDM04_N_per_type[key] / BDM04_total_area_hemisphere
            count = count + neurons * synapses_per_neuron[key]
        cat_counts.append(count)

    import matplotlib.pyplot as plt
    plt.plot(monkey_counts)
    plt.plot(cat_counts)
    plt.legend(['monkey', 'cat'])
    plt.show()


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
    'V6': 7.7, # Galletti et al. (1999)
    'MSTd': 16.0, # Komatsu et al. (1988)
    'AIP': None, # but see Romero & Janssen (2016)
    'TEO': 8.9, # Boussaoud et al. (1991)
    'TEpd': 38.5 # for TE generally in 0-10deg range, from Boussaoud et al. (1991), but sites look fairly dorsal (Fig 4)
}

def get_RF_size(area):
    result = None
    if area in RF_diameter_5_degrees_eccentricity.keys():
        result = RF_diameter_5_degrees_eccentricity[area]
    return result

class Yerkes19:
    """
    TODO
    """

    def __init__(self):
        folder = data_folder()

        midthickness_data = nibabel.load(folder + '/donahue/MacaqueYerkes19.R.midthickness.32k_fs_LR.surf.gii')
        self.midthickness_points \
            = np.array(midthickness_data.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data)
        self.triangles \
            = np.array(midthickness_data.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data)

        very_inflated_data = nibabel.load(folder + '/donahue/MacaqueYerkes19.R.very_inflated.32k_fs_LR.surf.gii')
        self.very_inflated_points \
            = np.array(very_inflated_data.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data)

        #TODO: make this automatically
        label_tree = ET.parse(folder + '/donahue/MarkovCC12_M132_91-area.32k_fs_LR.dlabel.xml')
        root = label_tree.getroot()

        self.areas = []
        for label_name in root.findall('Matrix/MatrixIndicesMap/NamedMap/LabelTable/Label'):
            # print(label.attrib)
            self.areas.append(label_name.text)

        point_inds = []
        for brain_model in root.findall('Matrix/MatrixIndicesMap/BrainModel'):
            if brain_model.attrib['BrainStructure'] == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
                vi = brain_model.find('VertexIndices')
                for s in vi.text.split():
                    point_inds.append(int(s))
        self.point_inds = np.array(point_inds)

        label_data = nibabel.load(folder + '/donahue/MarkovCC12_M132_91-area.32k_fs_LR.dlabel.nii')
        self.point_area_inds = np.array(label_data.get_data()).flatten()[:len(self.point_inds)]
        self._assign_triangles()

    def _assign_triangles(self):
        expanded_point_area_inds = np.zeros(np.max(self.point_inds)+1)
        for i in range(len(self.point_inds)):
            expanded_point_area_inds[self.point_inds[i]] = self.point_area_inds[i]

        self.triangle_area_inds = np.zeros(self.triangles.shape[0]) #unassigned
        for i in range(self.triangles.shape[0]):
            vertex_area_inds = np.zeros(3)
            for j in range(3):
                vertex_area_inds[j] = expanded_point_area_inds[self.triangles[i,j]]

            if vertex_area_inds[0] == vertex_area_inds[1] or vertex_area_inds[0] == vertex_area_inds[2]:
                self.triangle_area_inds[i] = vertex_area_inds[0]
            elif vertex_area_inds[1] == vertex_area_inds[2]:
                self.triangle_area_inds[i] = vertex_area_inds[1]

    def get_points_in_area(self, area, inflated=False):
        assert area in self.areas, '%s is not in the list of cortical areas' % area

        area_ind = self.areas.index(area)
        point_inds = [self.point_inds[i] for i, x in enumerate(self.point_area_inds) if x == area_ind]

        if inflated:
            points = np.array([self.very_inflated_points[i] for i in point_inds])
        else:
            points = np.array([self.midthickness_points[i] for i in point_inds])

        return points

    def get_surface_area_total(self):
        surface_area = 0

        for triangle in self.triangles:
            a = self.midthickness_points[triangle[0]]
            b = self.midthickness_points[triangle[1]]
            c = self.midthickness_points[triangle[2]]
            surface_area = surface_area + _get_triangle_area(a, b, c)

        return surface_area

    def get_surface_area(self, area):
        assert area in self.areas, '%s is not in the list of cortical areas' % area

        area_ind = self.areas.index(area)

        surface_area = 0
        for i in range(self.triangles.shape[0]):
            if self.triangle_area_inds[i] == area_ind:
                triangle = self.triangles[i]
                a = self.midthickness_points[triangle[0]]
                b = self.midthickness_points[triangle[1]]
                c = self.midthickness_points[triangle[2]]
                surface_area = surface_area + _get_triangle_area(a, b, c)

        return surface_area

    def get_centre(self, area):
        points = self.get_points_in_area(area, inflated=False)
        return np.mean(points, axis=0)


def _get_triangle_area(a, b, c):
    v1 = b - a
    v2 = c - a
    return np.linalg.norm(np.cross(v1, v2)) / 2.


yerkes19 = Yerkes19()
e07 = E07()


M14_FV91 = {
    # TEO is roughly PITd and PITv (Zeki, 1996), and it looks like Markov TEO injection may be in PITd.
    # On the other hand TEOm position relative to TEO is similar to PITd relative to PITv. To be conservative,
    # we'll map TEO to PITd and consider TEOm to lack a corresponding area.
    'TEO': 'PITd',
    'TEpd': 'CITd',
    'TEpv': 'CITv',
    'TEad': 'AITd',
    'TEav': 'AITv',
    'MST': 'MSTd',  # no injection into MST, FLNe is sum across MSTl, MSTd
    '7A': '7a',
    '7B': '7b',
    'V6': 'PO',  # this isn't a great correspondence; see Shipp et al. 2001
}


def map_M14_to_FV91(area):
    """
    :param area: Area name from Markov et al. parcellation
    :return: Name of most similar area from CoCoMac
    """

    if area in M14_FV91.keys():
        area = M14_FV91[area]

    return area


def map_FV91_to_M14(area):
    """
    :param area: Area name from CoCoMac
    :return: Name of most similar area from Markov et al.
    """

    FV91_M14 = dict((M14_FV91[key], key) for key in M14_FV91)

    if area in FV91_M14.keys():
        area = FV91_M14[area]

    return area


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


    @staticmethod
    def _print_fraction_asymmetric(layer):
        as_col_num = 14
        asymmetric_synapses_per_neuron = _get_synapses_per_layer_cat_V1(layer)[:,as_col_num]

        neurons = []
        for target in BDM04_targets:
            neurons.append(BDM04_N_per_type[target])

        asymmetric_synapses_per_type = np.multiply(asymmetric_synapses_per_neuron, neurons)

        layers = []
        asymmetric_synapses_per_layer = []
        for layer in BDM04_excitatory_types.keys():
            if layer != 'extrinsic':
                sum = 0
                for type in BDM04_excitatory_types[layer]:
                    sum = sum + asymmetric_synapses_per_type[BDM04_targets.index(type)]

                layers.append(layer)
                asymmetric_synapses_per_layer.append(sum)

        total = np.sum(asymmetric_synapses_per_layer)
        for i in range(5):
            # layer = BDM04_excitatory_types.keys()[i]
            fraction = asymmetric_synapses_per_layer[i] / total
            print('Cell layer: {} excitatory synapses: {}%'.format(layers[i], int(round(fraction*100))))

"""
We constrain the FLNe only for connections in Markov et al. (2012). Reasonable constraints could be 
added for other connections in several ways:
 
1) Other quantitative datasets may provide some additional coverage, e.g. visual prefrontal afferents 
data in Table 2 of Hilgetag, Medalla, Beul, and Barbas, “The primate connectome in context: Principles 
of connections of the cortical visual system,” Neuroimage, vol. 134, pp. 685–702, 2016.

2) The distance rule of Ercsey-Ravasz, Markov, Lamy, VanEssen, Knoblauch, Toroczkai, 
and Kennedy, “A Predictive Network Model of Cerebral Cortical Connectivity Based on a Distance Rule,” 
Neuron, vol. 80, no. 1, pp. 184–197, 2013. However this requires tracing the shortest path through white matter. 
(The authors seem to have used this software to do so: http://exploranova.info/m9732/) Also, while there is a 
correlation, there is also plenty of spread (see their Figure 2A). 

3) Hilgetag et al. (2016) suggest an alternative rule based on differences in cell density, and give a table of 
densities for a number of areas. The correlation seems good and this is simpler to do than (2) but there are 
limitations. First, they have only shown the correlation for their small "prefrontal visual afferents" dataset. 
Second, they don't give densities for all areas. This could be explored further by testing the correlation on the 
Markov et al. 29x29 matrix with available cell densities. If it looks promising, other densities could be 
guessed based on curve fit in D. J. Cahalane, C. J. Charvet, and B. L. Finlay, “Systematic, balancing gradients 
in neuron density and number across the primate isocortex,” Front. Neuroanat., vol. 6, no. July, pp. 1–12, 2012.
Or a Yerkes19 flatmap could be mapped to the flatmap in C. E. Collins, D. C. Airey, N. a Young, D. B. Leitch, and 
J. H. Kaas, “Neuron densities vary across and within cortical areas in primates.,” Proc. Natl. Acad. Sci. U. S. A., 
vol. 107, no. 36, pp. 15927–32, Sep. 2010.
"""


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

    def get_sources_with_fallback(self, target):
        if self.is_injection_site(target):
            return self.get_sources(target)
        else:
            c = CoCoMac()
            sources = c.get_source_areas(map_M14_to_FV91(target))
            sources = [map_FV91_to_M14(source) for source in sources]
            return [source for source in sources if source in yerkes19.areas]

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
            # calculate estimate from regression with inter-area distance
            # y = Yerkes19()
            source_centre = yerkes19.get_centre(source)
            target_centre = yerkes19.get_centre(target)
            distance = np.linalg.norm(source_centre - target_centre)
            log_FLNe = - 4.38850006 - 0.11039442*distance
            return np.exp(log_FLNe)

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


def get_areas():
    return yerkes19.areas


def get_layers(area):
    # Schmidt et al. (2018) say that TH lacks L4 but don't give a reference. However
    # Felleman & Van Essen (1991) say that several connections to TH terminate on L4
    # (F pattern in their Table 5).

    if area == 'V1':
        return ['1', '2/3', '3B', '4A', '4B', '4Calpha', '4Cbeta', '5', '6']
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


def _get_thickness_V2(layer):
    """
    :param layer: layer name
    :return: Average layer thickness from two examples (Balaram et al. and Shepherd), rescaled
        so that V1 examples from the same sources match population averages from O'Kusky et al.
    """
    assert layer in get_layers('V2')

    total_thickness_V1 = _total_thickness(OC82_thickness_V1)
    total_thickness_V1_BYK = _total_thickness(BYK14_thickness_V1)
    total_thickness_V1_S = _total_thickness(S03_thickness_V1)

    thickness_BYK = BYK14_thickness_V2[layer] * total_thickness_V1 / total_thickness_V1_BYK
    thickness_S = S03_thickness_V2[layer] * total_thickness_V1 / total_thickness_V1_S

    return (thickness_BYK + thickness_S) / 2


def _get_neurons_per_mm3_V1(layer):
    pass


def _get_neurons_per_mm2_V1(layer):
    """
    :param layer: Name of a V1 layer.
    :return: Neurons per square mm of cortical surface in layer.
    """
    assert layer in get_layers('V1')

    n23 = OC82_n_neurons_per_mm2_V1['2/3-1'] \
          + OC82_n_neurons_per_mm2_V1['2/3-2'] \
          + OC82_n_neurons_per_mm2_V1['2/3-3']

    # we will divide layer 2/3 according to thicknesses in Balaram et al.
    t2 = BYK14_thickness_V1['2']
    t3A = BYK14_thickness_V1['3A']
    t3B = BYK14_thickness_V1['3B']

    if layer == '2/3':
        result = round(n23 * (t2 + t3A) / (t2 + t3A + t3B))
    elif layer == '3B':
        result = round(n23 * t3B / (t2 + t3A + t3B))
    else:
        result = 0
        for key in OC82_n_neurons_per_mm2_V1.keys():
            if key.startswith(layer):
                result = result + OC82_n_neurons_per_mm2_V1[key]

    return result


def _get_neurons_per_mm2_V2(layer):
    """
    :param layer: Name of V2 layer
    :return: Estimate of neurons per square mm of cortical surface in layer. This is based on
        V1 estimates, scaled by relative thickness of the corresponding V2 layer.
        Following Balaram et al. (see also refs therein) we group 4A and 4B with layer 3 for this
        purpose. This is probably not highly accurate, as cells are particularly small in V1
        (see refs in Collins, C. E., Airey, D. C., Young, N. A., Leitch, D. B., & Kaas, J. H. (2010).
        Neuron densities vary across and within cortical areas in primates. PNAS, 107(36), 15927-15932.)
        #TODO: fix this using Carlo & Stevens and Srinivasan, Carlo & Stevens
    """
    result = None
    if layer == '2/3':
        sublayers = ['2/3', '3B', '4A', '4B']

        n_V1 = 0
        thickness_V1 = 0
        for sublayer in sublayers:
            n_V1 = n_V1 + _get_neurons_per_mm2_V1(sublayer)
            thickness_V1 = thickness_V1 + _get_thickness_V1(sublayer)

        # print('thickness V2 {} V1 {} #V1 {}'.format(_get_thickness_V2('2/3'), thickness_V1, n_V1))
        result = n_V1 * _get_thickness_V2('2/3') / thickness_V1
    elif layer == '4':
        sublayers = ['4Calpha', '4Cbeta']

        n_V1 = 0
        thickness_V1 = 0
        for sublayer in sublayers:
            n_V1 = n_V1 + _get_neurons_per_mm2_V1(sublayer)
            thickness_V1 = thickness_V1 + _get_thickness_V1(sublayer)

        result = n_V1 * _get_thickness_V2('2/3') / thickness_V1
    else:
        result = _get_neurons_per_mm2_V1(layer) * _get_thickness_V2(layer) / _get_thickness_V1(layer)

    return result


# def get_num_neurons(area, layer):
#     #TODO: replace with Schmidt et al.
#     """
#     Cortical thickness and cell density are negatively correlated in visual cortex in many primate
#     species:
#
#     la Fougère, C., Grant, S., Kostikov, A., Schirrmacher, R., Gravel, P., Schipper, H. M., ... &
#     Thiel, A. (2011). Where in-vivo imaging meets cytoarchitectonics: the relationship between cortical
#     thickness and neuronal density measured with high-resolution [18 F] flumazenil-PET. Neuroimage, 56(3), 951-960.
#
#     Cahalane, D. J., Charvet, C. J., & Finlay, B. L. (2012). Systematic, balancing gradients in neuron density
#     and number across the primate isocortex. Frontiers in neuroanatomy, 6.
#
#     TODO: docs
#     :param area:
#     :param layer:
#     :return:
#
#     """
#     # if yerkes19 is None:
#     #     yerkes19 = Yerkes19()
#
#     surface_area = yerkes19.get_surface_area(area)
#
#     if area == 'V1':
#         density = _get_neurons_per_mm2_V1(layer)
#     else:
#         density = _get_neurons_per_mm2_V2(layer)
#
#     # We multiply by 0.75 to match fraction excitatory cells; see Hendry et al. (1987) J Neurosci
#     return int(0.75 * surface_area * density)


def calculate_mean_ascending_SLN():
    """
    Calculates mean %SLN for connections that ascend the FV91 hierarchy.
    This mean can be used as a fallback for connections not characterized
    by Markov et al.

    The result is 65.42%
    """

    sources, targets, percents = _read_supragranular_layers_percent()
    d = []
    p = []
    y = Yerkes19()

    for i in range(len(sources)):
        if sources[i] in y.areas and targets[i] in y.areas:
            s = map_M14_to_FV91(sources[i])
            t = map_M14_to_FV91(targets[i])
            if s in FV91_hierarchy and t in FV91_hierarchy and FV91_hierarchy[t] > FV91_hierarchy[s]:
                source_centre = y.get_centre(sources[i])
                target_centre = y.get_centre(targets[i])
                distance = np.linalg.norm(source_centre - target_centre)
                d.append(distance)
                p.append(percents[i])
                print('{}->{}: {}'.format(sources[i], targets[i], distance))

    print('Mean %SLN: {}'.format(np.mean(p)))

    import matplotlib.pyplot as plt
    plt.scatter(d, p)
    plt.title('Not much correlation between %SLN and inter-area distance')
    plt.show()


def calculate_flne_vs_distance():
    """
    Result:
    Correlation between distance and log(FLNe): -0.4464287686487642. Regression slope: -0.11039442; bias: -4.38850006
    """
    import pickle

    # sources, targets, fractions = _read_fraction_labelled_neurons_extrinsic()
    # y = Yerkes19()
    # d = []
    # f = []
    #
    # for i in range(len(sources)):
    #     if sources[i] in y.areas and targets[i] in y.areas:
    #         source_centre = y.get_centre(sources[i])
    #         target_centre = y.get_centre(targets[i])
    #         distance = np.linalg.norm(source_centre - target_centre)
    #         d.append(distance)
    #         f.append(fractions[i])
    #         print('{}->{}: {}'.format(sources[i], targets[i], distance))
    #
    # with open('flne-and-distance.pkl', 'wb') as file:
    #     pickle.dump((d, f), file)

    with open('flne-and-distance.pkl', 'rb') as file:
        (d, f) = pickle.load(file)

    import matplotlib.pyplot as plt

    f = np.array(f)
    d = np.array(d)

    a = np.ones((len(f),2))
    a[:, 0] = d

    r = np.corrcoef(d, np.log(f))

    x, res, rank, s = np.linalg.lstsq(a, np.log(f))
    approx = np.matmul(a, x)

    print('Correlation between distance and log(FLNe): {}. Regression coefficients (slope; bias): {}'.format(r[0, 1], x))

    plt.plot(d, np.log(f), '.')
    plt.plot(d, approx, 'k.')
    plt.xlabel('inter-area distance (mm)')
    plt.ylabel('log(FLNe)')
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = Data()

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

    iac = InterAreaConnections()
    plt.imshow(iac.get_connectivity_grid())
    plt.show()

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
    # plot_spine_count_estimates()

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
