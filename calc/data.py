"""
TODO:
- M132 parcellation names
- cell densities / mm^3 (Beul et al. figure)
- RF sizes for LGN, V1, V2, V4, PIT, MT, MST
- FLNe (from excel file)
- SLN (from excel file)
- CoCoMac layer-wise codes
- inflated positions
"""

#TODO: refs for cell density and thickness, use V2 data due to negative correlation
#TODO: Balaram and Shepherd total thicknesses don't match O'Kusky means: scale them and average V2

# Sincich, L. C., Adams, D. L., & Horton, J. C. (2003). Complete flatmounting of the macaque cerebral cortex. Visual neuroscience, 20(6), 663-686.
mean_cortical_area = 10430

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
        purpose.
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

    for layer in get_layers('V2'):
        print(_get_neurons_per_mm2_V2(layer))

