# Example systems

# TODO: looks like bug in layer-5 # units outside v1 and v2?
# TODO: fraction of input to L4 V1 from LGN vs # extrinsic inputs

from calc.system import System
from calc.stride import StridePattern
from calc.data import Data
from calc.optimization import test_stride_pattern


data = Data()

def add_areas(system, cortical_areas):
    for area in cortical_areas:
        layers = data.get_layers(area)
        for layer in layers:
            if layer != '1' and layer != '6':
                name = _pop_name(area, layer)

                n = data.get_num_neurons(area, layer)
                e = data.get_extrinsic_inputs(area, layer)
                w = data.get_receptive_field_size(area) if layer == '2/3' else None

                system.add(name, n, e, w)

        _add_intrinsic_forward_connections(system, area)


def connect_areas(system, cortical_areas):
    for target in cortical_areas:
        for source in data.get_source_areas(target, feedforward_only=True):
            if source in cortical_areas:
                FLNe = data.get_FLNe(source, target)
                SLN = data.get_SLN(source, target)

                supra_source_pop = '{}_2/3'.format(source)
                infra_source_pop = '{}_5'.format(source)
                target_pop = '{}_4'.format(target)
                system.connect_areas(supra_source_pop, target_pop, FLNe*SLN/100)
                system.connect_areas(infra_source_pop, target_pop, FLNe*(1-SLN/100))


def connect_areas_in_streams(system, cortical_areas):
    """
    Dorsal / ventral aware version. Include V1 and V2 in cortical_areas, but they must be
    connected to each other separately.
    """
    for target in [a for a in cortical_areas if a not in ('V1', 'V2')]:
        for source in data.get_source_areas(target, feedforward_only=True):
            # print(source)
            if source in cortical_areas:
                FLNe = data.get_FLNe(source, target)
                SLN = data.get_SLN(source, target)
                # print('{}->{} FLNe: {} SLN: {}'.format(source, target, FLNe, SLN))
                # FLN, SLF, TLF = get_connection_details(source, target)

                target_pop = '{}_4'.format(target)

                if source == 'V1':
                    if is_ventral(target):
                        supra_source_pop = 'V1_2/3'  # true for V4, not sure about others
                        infra_source_pop = 'V1_5'
                        system.connect_areas(supra_source_pop, target_pop, FLNe*SLN/100)
                        system.connect_areas(infra_source_pop, target_pop, FLNe*(1-SLN/100))
                    else:
                        supra_source_pop = 'V1_4B'  # true for MT, not sure about others
                        system.connect_areas(supra_source_pop, target_pop, FLNe*SLN/100)
                elif source == 'V2':
                    if is_ventral(target):
                        supra_source_pop = 'V2thin_2/3'
                        infra_source_pop = 'V2thin_5'
                    else:
                        supra_source_pop = 'V2thick_2/3'
                        infra_source_pop = 'V2thick_5'
                    system.connect_areas(supra_source_pop, target_pop, FLNe * SLN / 100)
                    system.connect_areas(infra_source_pop, target_pop, FLNe * (1 - SLN / 100))
                else:
                    supra_source_pop = '{}_2/3'.format(source)
                    infra_source_pop = '{}_5'.format(source)
                    system.connect_areas(supra_source_pop, target_pop, FLNe*SLN/100)
                    system.connect_areas(infra_source_pop, target_pop, FLNe*(1-SLN/100))


def is_ventral(area):
    # see Schmidt et al. (2018) Fig 7 re. V4t
    return area in ['V4', 'VOT', 'PITd', 'PITv', 'CITd', 'CITv', 'AITd', 'AITv', 'TF', 'TH']


def make_big_system():
    # complications:
    # - AIP connections studied by Borra but not in CoCoMac
    # - CIP should maybe be split from LIP (or related to PIP?)
    # - pulvinar connectivity
    # - how to normalize FLNe
    # V6, V6A: should really use Shipp et al. 2001 rather than CoCoMac, but map to PO for now

    system = System()
    w_rf_0 = .2
    system.add_input(750000, w_rf_0)

    # Estimates of LGN cell numbers (one side) from:
    # Weber, Arthur J., et al. (2000) Experimental glaucoma and cell size, density, and number in the primate lateral
    # geniculate nucleus. Investigative ophthalmology & visual science 41.6: 1370 - 1379.
    n_LGN = 1270000
    n_magno_LGN = .103 * n_LGN
    n_parvo_LGN = .897 * n_LGN

    # Weber et al. say, "For normal animals, the total number of neurons in the LGN was estimated to be
    # approximately 1.27 million, with 10.3% located in the M-layers and 89.7% in the P-layers." Koniocellular
    # neurons are between these layers, and similar in number to M cells, so we'll add another 10.3%
    n_konio_LGN = .103 * n_LGN

    # Guessing convergence from RGC to LGN based on comments page 54 of:
    # Lee, B.B., Virsu, V., & Creutzfeldt, O.D.(1983).Linear signal transmission from prepotentials to cells in the
    # macaqie lateral geniculate nucleus. Experimental Brain Research, 52(1), 50 - 56.
    convergence_LGN = 5

    # See also re. parallel projections from retina to LGN:
    # Leventhal, A.G., Rodieck, R.W., & Dreher, B.(1981).Retinal ganglion cell classes in the Old World
    # monkey: morphology and central projections. Science, 213(4512), 1139 - 1142.

    # Magno cells have slightly larger RFs than parvo:
    # Derrington, A.M., & Lennie, P.(1984).Spatial and temporal contrast sensitivities of neurones in lateral
    # geniculate nucleus of macaque.The Journal of Physiology, 357(1), 219 - 240.

    # Pixels correspond roughly to retinal ganglion cells
    # Setting LGN RF sizes similar to input (one-pixel kernels)
    system.add('parvo_LGN', n_parvo_LGN, 5, 1.041*w_rf_0)
    system.add('magno_LGN', n_magno_LGN, 5, 1.155*w_rf_0)
    system.add('konio_LGN', n_konio_LGN, 5, 1.155*w_rf_0)  #RF sizes highly scattered but not comparable to Magno (Xu et al., 2004, J Physiol)
    system.connect_areas(system.input_name, 'parvo_LGN', 1.)
    system.connect_areas(system.input_name, 'magno_LGN', 1.)
    system.connect_areas(system.input_name, 'konio_LGN', 1.)

    for layer in ['4Calpha', '4Cbeta', '4B', '2/3', '5']:
        n = data.get_num_neurons('V1', layer)
        e = data.get_extrinsic_inputs('V1', '4' if layer[0] == '4' else layer)
        if layer == '2/3':
            w = data.get_receptive_field_size('V1')
        elif layer == '4B':
            w = 1.1 * data.get_receptive_field_size('V1')  # TODO: get better estimate from Gilbert
        else:
            w = None
        system.add('V1_{}'.format(layer), n, e, w)

    system.connect_areas('parvo_LGN', 'V1_4Cbeta', 1.)
    system.connect_areas('magno_LGN', 'V1_4Calpha', 1.)
    system.connect_areas('konio_LGN', 'V1_2/3', 1.) #TODO: also 4A, but not sure about output of 4A
    system.connect_layers('V1_4Calpha', 'V1_4B', data.get_inputs_per_neuron('V1', '4', '2/3'))
    system.connect_layers('V1_4Cbeta', 'V1_2/3', data.get_inputs_per_neuron('V1', '4', '2/3'))
    system.connect_layers('V1_2/3', 'V1_5', data.get_inputs_per_neuron('V1', '2/3', '5'))
    system.connect_layers('V1_4Cbeta', 'V1_5', data.get_inputs_per_neuron('V1', '4', '5'))

    #TODO: use cell counts for V1 directly

    for area in ['V2thick', 'V2thin']:
        for layer in ['2/3', '4', '5']:
            name = _pop_name(area, layer)

            n = data.get_num_neurons('V2', layer) / 2  # dividing V2 equally into thick and thin+inter stripes
            e = data.get_extrinsic_inputs('V2', layer)
            w = data.get_receptive_field_size('V2') if layer == '2/3' else None

            system.add(name, n, e, w)

        system.connect_layers(_pop_name(area, '4'), _pop_name(area, '2/3'), data.get_inputs_per_neuron('V2', '4', '2/3'))
        system.connect_layers(_pop_name(area, '2/3'), _pop_name(area, '5'), data.get_inputs_per_neuron('V2', '2/3', '5'))
        system.connect_layers(_pop_name(area, '4'), _pop_name(area, '5'), data.get_inputs_per_neuron('V2', '4', '5'))

    # There is substantial forward projection from V1_5 to V2 and V3, also from V2_5 to V3. We assume V1_5 is
    # parvo, and V2_5 is separated into stripes. Will omit parvo, V1_5, and V2thin_5 projections to dorsal areas,
    # consistent with MT supression following magno inactivation, reported in:
    # Maunsell, J. H., Nealey, T. A., & DePriest, D. D. (1990). Magnocellular and parvocellular contributions to
    # responses in the middle temporal visual area (MT) of the macaque monkey. Journal of Neuroscience, 10(10), 3323-3334.
    FLNe = data.get_FLNe('V1', 'V2')
    SLN = data.get_SLN('V1', 'V2')
    system.connect_areas('V1_2/3', 'V2thin_4', FLNe * SLN / 100)
    system.connect_areas('V1_5', 'V2thin_4', FLNe * (1 - SLN / 100))
    system.connect_areas('V1_4B', 'V2thick_4', FLNe * SLN / 100)

    # Dorsal areas get connections from V1_4B and V2thick; ventral from V1_2/3, V1_5, and V2_thin
    # References on segregation in V2:
    # Shipp, S., & Zeki, S. (1985). Segregation of pathways leading from area V2 to areas V4 and V5 of
    # macaque monkey visual cortex. Nature, 315(6017), 322.
    # DeYoe, E. A., & Van Essen, D. C. (1985). Segregation of efferent connections and receptive field
    # properties in visual area V2 of the macaque. Nature, 317(6032), 58.
    # Sincich, L. C., & Horton, J. C. (2002). Divided by cytochrome oxidase: a map of the projections
    # from V1 to V2 in macaques. Science, 295(5560), 1734-1737.
    # DeYoe, E. A., & Van Essen, D. C. (1988). Concurrent processing streams in monkey visual cortex.
    # Trends in neurosciences, 11(5), 219-226.

    cortical_areas = data.get_areas()[:]
    cortical_areas.remove('MDP') # MDP has no inputs in CoCoMac
    # cortical_areas = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'MSTd', 'DP',
    #               'VIP', 'PITv', 'PITd', 'MSTl', 'CITv', 'CITd', 'AITv', 'FST', '7a', 'AITd']

    add_areas(system, [a for a in cortical_areas if a not in ('V1', 'V2')])
    connect_areas_in_streams(system, cortical_areas)

    system.prune_FLNe()
    system.check_connected()

    return system


def _pop_name(area, layer):
    return '{}_{}'.format(area, layer)


def _add_intrinsic_forward_connections(system, area):
    if area == 'V1':
        system.connect_layers('V1_4Cbeta', 'V1_2/3', data.get_inputs_per_neuron(area, '4', '2/3'))
        system.connect_layers('V1_2/3', 'V1_5', data.get_inputs_per_neuron(area, '2/3', '5'))
        system.connect_layers('V1_4Cbeta', 'V1_5', data.get_inputs_per_neuron('V1', '4', '5'))
    else:
        system.connect_layers(_pop_name(area, '4'), _pop_name(area, '2/3'), data.get_inputs_per_neuron(area, '4', '2/3'))
        system.connect_layers(_pop_name(area, '2/3'), _pop_name(area, '5'), data.get_inputs_per_neuron(area, '2/3', '5'))
        system.connect_layers(_pop_name(area, '4'), _pop_name(area, '5'), data.get_inputs_per_neuron(area, '4', '5'))


def make_small_system(miniaturize=False):
    #TODO: add V2 stripe distinctions (extract methods for parvo and magno early vision?)
    cortical_areas = ['V1', 'V2', 'V4', 'VOT', 'PITv', 'PITd', 'CITv', 'CITd', 'AITv', 'AITd', 'TH']

    system = System()
    system.add_input(750000, .2)

    area = 'V1'
    for layer in ['2/3', '4Cbeta', '5']:
        name = _pop_name(area, layer)

        n = data.get_num_neurons(area, layer)
        e = data.get_extrinsic_inputs(area, '4' if layer == '4Cbeta' else layer)
        w = data.get_receptive_field_size(area) if layer == '2/3' else None

        system.add(name, n, e, w)

    system.connect_areas(system.input_name, 'V1_4Cbeta', 1.)
    system.connect_layers(_pop_name(area, '4Cbeta'), _pop_name(area, '2/3'), data.get_inputs_per_neuron(area, '4', '2/3'))
    system.connect_layers(_pop_name(area, '2/3'), _pop_name(area, '5'), data.get_inputs_per_neuron(area, '2/3', '5'))
    system.connect_layers(_pop_name(area, '4Cbeta'), _pop_name(area, '5'), data.get_inputs_per_neuron(area, '4', '5'))

    add_areas(system, [area for area in cortical_areas if area != 'V1'])
    connect_areas(system, cortical_areas)

    system.prune_FLNe()
    system.check_connected()

    # graph = system.make_graph()
    # import networkx as nx
    # import matplotlib.pyplot as plt
    # from calc.system import get_layout
    # nx.draw_networkx(graph, pos=get_layout(system), arrows=True, font_size=10, node_size=1200, node_color='white')
    # plt.show()

    if miniaturize:
        for population in system.populations:
            if population.name != system.input_name:
                population.n = population.n / 20

    system.prune_FLNe()
    system.check_connected()
    return system


if __name__ == '__main__':
    # system = make_small_system(miniaturize=True)
    # system = make_big_system()
    # system.print_description()
    # net, training_curve = calc.optimization.test_stride_patterns(system, n=1)

    import pickle
    with open('stride-pattern-best-of-10.pkl', 'rb') as file:
        data = pickle.load(file)

    # data['system'].print_description()
    net, training_curve = test_stride_pattern(data['system'], data['strides'])
    with open('optimization-result.pkl', 'wb') as file:
        pickle.dump({'net': net, 'training_curve': training_curve}, file)

    print('**********************')
    print(net)
    print(training_curve)
