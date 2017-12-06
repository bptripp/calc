import nibabel
import numpy as np
import xml.etree.ElementTree as ET

"""
TODO:
- RF sizes for LGN, V1, V2, V4, PIT, MT, MST
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


def check_data():
    """
    Some data files must be downloaded separately from this repository. This function checks whether
    they are present in the expected form.
    """
    pass


def get_areas():
    #TODO: read these from a data file
    return ['V1', 'V2', 'V3', 'V4', 'MT']


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
    area = None

    if area == 'V1':
        density = _get_neurons_per_mm2_V1(layer)
    else:
        density = _get_neurons_per_mm2_V2(layer)

    return area * density


class Yerkes19:
    """
    TODO
    """

    def __init__(self):
        midthickness_data = nibabel.load('data/donahue/MacaqueYerkes19.R.midthickness.32k_fs_LR.surf.gii')
        self.midthickness_points \
            = np.array(midthickness_data.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data)
        self.midthickness_triangles \
            = np.array(midthickness_data.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data)

        very_inflated_data = nibabel.load('data/donahue/MacaqueYerkes19.R.very_inflated.32k_fs_LR.surf.gii')
        self.very_inflated_points \
            = np.array(very_inflated_data.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data)

        #TODO: make this automatically
        label_tree = ET.parse('data/donahue/MarkovCC12_M132_91-area.32k_fs_LR.dlabel.xml')
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

        label_data = nibabel.load('data/donahue/MarkovCC12_M132_91-area.32k_fs_LR.dlabel.nii')
        self.point_area_inds = np.array(label_data.get_data()).flatten()[:len(self.point_inds)]


    def get_points_in_area(self, area):
        assert area in self.areas, '%s is not in the list of cortical areas' % area

        print('+++')
        print(self.point_inds.shape)
        print(max(self.point_inds))
        print(self.point_area_inds.shape)
        print(max(self.point_area_inds))

        #TODO: clean this up
        area_ind = self.areas.index(area)
        point_inds = [self.point_inds[i] for i, x in enumerate(self.point_area_inds) if x == area_ind]
        points = np.array([self.very_inflated_points[i] for i in point_inds])
        # points = np.array([self.very_inflated_points[i] for i in point_inds])
        print(points.shape)
        return points

    def get_area_centre(self, area):
        pass


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
    # foo2 = nibabel.load('/Users/bptripp/Downloads/Donahue_et_al_2016_Journal_of_Neuroscience_W336/data/MarkovCC12_M132_91-area.32k_fs_LR.dlabel.nii')
    # print('labels')
    # print(foo2.shape)
    # # foo3 = nibabel.load('/Users/bptripp/Downloads/Donahue_et_al_2016_Journal_of_Neuroscience_W336 2/data/Macaque.MedialWall.32k_fs_LR.dlabel.nii')
    # # print(foo3.shape)
    # print(dir(foo2))
    # # print(foo2.extra)
    # # print(foo2.get_header())
    # # print(max(foo2.get_data().flatten()))
    #
    # import xml.etree.ElementTree as ET
    # import numpy as np
    # tree = ET.parse('/Users/bptripp/code/calc/calc/data/donahue/MarkovCC12_M132_91-area.32k_fs_LR.dlabel.xml')
    # # print(tree)
    #
    # root = tree.getroot()
    #
    # # for label in root.findall('Matrix/MatrixIndicesMap/NamedMap/LabelTable/Label'):
    # #     print(label.attrib)
    # #     print(label.text)
    #
    # left_ind = []
    # for brain_model in root.findall('Matrix/MatrixIndicesMap/BrainModel'):
    #     if brain_model.attrib['BrainStructure'] == 'CIFTI_STRUCTURE_CORTEX_LEFT':
    #         vi = brain_model.find('VertexIndices')
    #         for s in vi.text.split():
    #             left_ind.append(int(s))
    # print(len(left_ind))
    # print(max(np.array(left_ind)))
    #
    # right_ind = []
    # for brain_model in root.findall('Matrix/MatrixIndicesMap/BrainModel'):
    #     if brain_model.attrib['BrainStructure'] == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
    #         vi = brain_model.find('VertexIndices')
    #         for s in vi.text.split():
    #             right_ind.append(int(s))
    # print(len(right_ind))
    # print(max(np.array(right_ind)))
    #
    # import matplotlib.pyplot as plt
    # plt.plot(left_ind)
    # plt.plot(right_ind)
    # plt.show()
    #

    import os
    print(os.getcwd())

    y19 = Yerkes19()
    points_MT = y19.get_points_in_area('MT')
    points_V1 = y19.get_points_in_area('V1')
    points_V2 = y19.get_points_in_area('V2')
    points_TEO = y19.get_points_in_area('TEO')


    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = y19.very_inflated_points[::10,:]
    ax.scatter(points[:,0], points[:,1], points[:,2], c='k', marker='.')
    ax.scatter(points_MT[:,0], points_MT[:,1], points_MT[:,2], c='b', marker='o')
    ax.scatter(points_TEO[:,0], points_TEO[:,1], points_TEO[:,2], c='r', marker='o')
    # ax.scatter(points_V1[:,0], points_V1[:,1]-2, points_V1[:,2], c='r', marker='.')
    # ax.scatter(points_V2[:,0], points_V2[:,1]+2, points_V2[:,2], c='g', marker='.')
    plt.show()

    # print(y19.midthickness_points.shape)
    # print(y19.midthickness_triangles.shape)
    # print(y19.very_inflated_points.shape)
    # print(y19.areas)
    # print(y19.label_inds.shape)
    # print(y19.labels.shape)
