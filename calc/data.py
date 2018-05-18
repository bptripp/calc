import nibabel
import numpy as np
import xml.etree.ElementTree as ET
import csv
import json

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
    'MDP': 5, 'PO': 5, 'MT': 5, 'V4t': 5, 'V4': 5,
    'DP': 6, 'VOT': 6,
    'VIP': 7, 'LIP': 7, 'MSTd': 7, 'MSTl': 7, 'FST': 7, 'PITd': 7, 'PITv': 7,
    '7b': 8, '7a': 8, 'FEF': 8, 'STPp': 8, 'CITd': 8, 'CITv': 8,
    'STPa': 9, 'AITd': 9, 'AITv': 9,
    '36': 10, '46': 10, 'TF': 10, 'TH': 10
}


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

    return numerator / denominator


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
    # print(monkey_cat_ratio)
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
    with open('data_files/BDM04-Supplementary.txt') as file:
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
# TODO: closer look at this
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
        midthickness_data = nibabel.load('data_files/donahue/MacaqueYerkes19.R.midthickness.32k_fs_LR.surf.gii')
        self.midthickness_points \
            = np.array(midthickness_data.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data)
        self.triangles \
            = np.array(midthickness_data.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data)

        very_inflated_data = nibabel.load('data_files/donahue/MacaqueYerkes19.R.very_inflated.32k_fs_LR.surf.gii')
        self.very_inflated_points \
            = np.array(very_inflated_data.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data)

        #TODO: make this automatically
        label_tree = ET.parse('data_files/donahue/MarkovCC12_M132_91-area.32k_fs_LR.dlabel.xml')
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

        label_data = nibabel.load('data_files/donahue/MarkovCC12_M132_91-area.32k_fs_LR.dlabel.nii')
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
        with open('data_files/cocomac/connectivity_matrix_layers.json') as file:
            self.layers = json.load(file)

        with open('data_files/cocomac/connectivity_matrix_densities.json') as file:
            self.densities = json.load(file)

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
            return [map_FV91_to_M14(source) for source in sources if source in yerkes19.areas]

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

    with open('data_files/markov/Cercor_2012 Table.csv') as csvfile:
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

    with open('data_files/markov/JCN_2013 Table.csv') as csvfile:
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

def get_num_neurons(area, layer):
    """
    Cortical thickness and cell density are negatively correlated in visual cortex in many primate
    species:

    la Fougère, C., Grant, S., Kostikov, A., Schirrmacher, R., Gravel, P., Schipper, H. M., ... &
    Thiel, A. (2011). Where in-vivo imaging meets cytoarchitectonics: the relationship between cortical
    thickness and neuronal density measured with high-resolution [18 F] flumazenil-PET. Neuroimage, 56(3), 951-960.

    Cahalane, D. J., Charvet, C. J., & Finlay, B. L. (2012). Systematic, balancing gradients in neuron density
    and number across the primate isocortex. Frontiers in neuroanatomy, 6.

    TODO: docs
    :param area:
    :param layer:
    :return:

    """
    # if yerkes19 is None:
    #     yerkes19 = Yerkes19()

    surface_area = yerkes19.get_surface_area(area)

    if area == 'V1':
        density = _get_neurons_per_mm2_V1(layer)
    else:
        density = _get_neurons_per_mm2_V2(layer)

    # We multiply by 0.75 to match fraction excitatory cells; see Hendry et al. (1987) J Neurosci
    return int(0.75 * surface_area * density)

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
    # print(get_num_neurons('V1', '2/3'))
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

    calculate_flne_vs_distance()

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
