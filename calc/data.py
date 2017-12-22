import nibabel
import numpy as np
import xml.etree.ElementTree as ET
import csv
import textract

"""
TODO:
- make example network with forward, backward, recurrent connections 
- test reasonableness of some layer neuron counts

- FLNe (from excel file)
- SLN (from excel file)
- CoCoMac layer-wise codes
"""

"""
Where possible, connection strengths and origin layers are taken from:

Markov, N. T., Ercsey-Ravasz, M. M., Ribeiro Gomes, A. R., Lamy, C., Magrou, L., Vezoli, 
J., ... & Sallet, J. (2012). A weighted and directed interareal connectivity matrix for macaque 
cerebral cortex. Cerebral cortex, 24(1), 17-36.

Additional connections and termination layers are taken from CoCoMac 2.0:
 
Bakker, R., Wachtler, T., & Diesmann, M. (2012). CoCoMac 2.0 and the future of tract-tracing 
databases. Frontiers in neuroinformatics, 6.

The strengths of additional connections are estimated according to the distance rule from:
 
Ercsey-Ravasz, M., Markov, N. T., Lamy, C., Van Essen, D. C., Knoblauch, K., Toroczkai, Z., 
& Kennedy, H. (2013). A predictive network model of cerebral cortical connectivity based on a 
distance rule. Neuron, 80(1), 184-197.

This is similar to the model in: 
Schmidt, M., Bakker, R., Shen, K., Bezgin, G., Hilgetag, C. C., Diesmann, M., & van Albada, S. J. (2015). 
Full-density multi-scale account of structure and dynamics of macaque visual cortex. 
arXiv preprint arXiv:1511.09364.

This code is vision-centric at the moment. Here are some relevant references for other parts of cortex:
 
Barbas, H., & Rempel-Clower, N. (1997). Cortical structure predicts the pattern of corticocortical 
connections. Cerebral cortex (New York, NY: 1991), 7(7), 635-646.

Dombrowski, S. M., Hilgetag, C. C., & Barbas, H. (2001). Quantitative architecture distinguishes 
prefrontal cortical systems in the rhesus monkey. Cerebral Cortex, 11(10), 975-988.

Dum, R. P., & Strick, P. L. (2005). Motor areas in the frontal lobe: the anatomical substrate for 
the central control of movement. Motor cortex in voluntary movements: a distributed system for 
distributed functions, 3-47. 

[1] G. N. Elston, “Cortical heterogeneity: Implications for visual processing and polysensory integration,” 
J. Neurocytol., vol. 31, no. 3–5 SPEC. ISS., pp. 317–335, 2002.
"""

#TODO: refs for cell density and thickness, use V2 data due to negative correlation
#TODO: Balaram and Shepherd total thicknesses don't match O'Kusky means: scale them and average V2

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

# Balaram, P., Young, N. A., & Kaas, J. H. (2014). Histological features of layers and sublayers in cortical visual areas V1 and V2 of chimpanzees, macaque monkeys, and humans. Eye and brain, 2014(6 Suppl 1), 5.
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

BDM04_neuron_types = [
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

# Extracted from Fig 9A via WebPlotDigitizer

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


def _get_synapses_per_layer(layer):
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

    print('cat neurons: {} synapses: {}'.format(neurons, synapses))
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

    print('monkey neurons: {} synapses: {}'.format(neurons, synapses))
    return synapses / neurons


#TODO: enter Binzegger connection tables
#TODO: scale layer-wise to monkey synapse densities
#TODO: calculate weighted-average in-degree by summing inputs across layers
#TODO: figure out FLNe equivalent -- or remove from cost?

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


def get_in_degree(layer):
    """
    Mean inbound connections per neuron, neglecting connections within layer. Only
    excitatory cells are considered, based on the rationale in:

    Parisien, C., Anderson, C. H., & Eliasmith, C. (2008). Solving the problem of negative
    synaptic weights in cortical models. Neural computation, 20(6), 1473-1494.

    We use inputs to excitatory cells, averaged over excitatory cell types (weighted by numbers
    of each cell type). This is based mainly on estimates by Binzegger et al. (2004) for cat V1.
    To rescale for macaque V1, we multiply by (monkey synapses per neuron) / (car synapses per
    neuron).

    :param layer:
    :return:
    """
    assert layer in ['1', '2/3', '4', '5', '6']

    if layer == '1':
        types = ['sp1']
    elif layer == '2/3':
        types = ['p2/3']
    elif layer == '4':
        types = ['ss4(L4)', 'ss4(L2/3)', 'p4']
    elif layer == '5':
        types = ['p5(L2/3)', 'p5(L5/6)']
    elif layer == '6':
        types = ['p6(L4)', 'p6(L5/6)']




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
    'V1': None,
    'V2': None,
    'V4': None,
    'MT': None,
    'MST': None,
    'AIP': None,
    'TEO': None,
    'TEav': None
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

    with open('data_files/markov/JCN_2013 Table.csv') as csvfile:
        r = csv.reader(csvfile)
        header_line = True
        for row in r:
            if header_line:
                header_line = False
            elif row[2] != 'NA' and len(row[2]) > 0:
                sources.append(row[0])
                targets.append(row[1])
                percents.append(float(row[2]))

    return sources, targets, percents


def get_sources(area):
    return []

def get_feedforward(source, target):
    #TODO: implement
    # if area != 'V1' and TLF[3] > .5:  # we only want feedforward connections before optimization
    return True

def get_connection_details(source, target):
    #TODO: implement
    #TODO: distance rule
    #TODO: docs for CoCoMac files
    #TODO: note this won't respect specificity of connections from layer X of A to layer Y of B
    #   (not sure if this is known)
    fraction_labelled_neurons = None
    source_layer_fractions = []
    target_layer_fractions = []

    return fraction_labelled_neurons, source_layer_fractions, target_layer_fractions


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

    return int(surface_area * density)

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
    print(_get_synapses_per_layer('6'))
