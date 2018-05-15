# Example systems

# TODO
# - should I set only feedforward connections prior to optimization?
# - interaction between f for inter-area and inter-laminar connections
# - magno and parvocellular LGN, V1, V2, V4, MT, MST, LIP, TEO, TE
# - use SLN from Markov, but how to fill in for missing areas? CoCoMac I guess
# - should only enforce FLNe across layers if SLN available, since CoCoMac is rarely quantitative
#   (requires a model change in system)

# TODO: looks like bug in layer-5 # units outside v1 and v2?

from calc.system import System
from calc.data import get_layers, get_num_neurons, get_RF_size, synapses_per_neuron, get_areas, synapses_per_neuron, Markov, CoCoMac
# from calc.data import get_sources, get_feedforward, get_connection_details
import calc.data
import calc.conversion


def make_system(cortical_areas):
    # cortical_areas = ['V1', 'V2', 'V3', 'V3A', 'V4', 'MT', 'MST', 'VIP', 'LIP', 'TEO', 'TEpd']

    system = System()

    #TODO: problem: for V1 I have to divide inputs by sublayer
    #TODO: this is a hack
    # return ['1', '2/3', '3B', '4A', '4B', '4Calpha', '4Cbeta', '5', '6']
    layer_map = {
        '1': '1',
        '2/3': '2/3',
        '3B': '2/3',
        '4': '4',
        '4A': '4',
        '4B': '4',
        '4Calpha': '4',
        '4Cbeta': '4',
        '5': '5',
        '6': '6'
    }

    system.add_input(750000, .2)
    # system.add(system.input_name, 750000, 0, .2)

    #TODO: this is parvocellular

    for area in cortical_areas:
        layers = get_layers(area)
        for layer in layers:
            if layer != '1' and layer != '6' and layer != '4Calpha' and layer != '4A' and layer != '4B' and layer != '3B':
                name = _pop_name(area, layer)
                n = get_num_neurons(area, layer)
                e = synapses_per_neuron(area, 'extrinsic', layer_map[layer])

                # scale different layers differently; Gilbert 1977, Fig 8 is a source
                # but using more conservative numbers that prevent 0-width kernels
                rf_size = get_RF_size(area)
                if rf_size is None:
                    w = None
                else:
                    if layer == '2/3' or layer.startswith('3'):
                        w = get_RF_size(area)
                    # elif layer.startswith('4'):
                    #     w = 0.8*get_RF_size(area)
                    # elif layer == '5':
                    #     w = 1.2*get_RF_size(area)
                    # elif layer == '6':
                    #     w = 1.4*get_RF_size(area)
                    else:
                        w = None
                system.add(name, n, e, w)

        _add_intrinsic_forward_connections(system, area)

    system.print_description()
    system.connect_areas(system.input_name, 'V1_4Cbeta', 1.)

    m = Markov()
    for target in cortical_areas:
        for source in m.get_sources(target):
            # print(source)
            if source in cortical_areas and get_feedforward(source, target):
                FLNe = m.get_FLNe(source, target)
                SLN = m.get_SLN(source, target)
                print('{}->{} FLNe: {} SLN: {}'.format(source, target, FLNe, SLN))
                # FLN, SLF, TLF = get_connection_details(source, target)

                supra_source_pop = '{}_2/3'.format(source)
                infra_source_pop = '{}_5'.format(source)
                target_pop = '{}_4'.format(target)
                system.connect_areas(supra_source_pop, target_pop, FLNe*SLN/100)
                system.connect_areas(infra_source_pop, target_pop, FLNe*(1-SLN/100))

    """
    FLNe works straightforwardly as long as input is to layer 4. 
    Maybe I force it to be this way? and adjust in-degree for each layer 
    to only include feedforward connections (add others layer).
    """
    return system


def get_feedforward(source, target):
    source = CoCoMac._map_M14_to_FV91(source)
    target = CoCoMac._map_M14_to_FV91(target)
    return calc.data.FV91_hierarchy[source] < calc.data.FV91_hierarchy[target]


def _add_early_visual_system(system):
    #TODO: read fuzzy logic of thamamic connectivity
    #TODO: not sure about sizes of LGN layers, numbers of magno vs parvo LGN and RGC cells
    #TODO: not sure about in-degree of LGN cells
    #TODO: do magno V1 cells have larger RFs than parvo?

    # RGCs and lateral geniculate nucleus have magnocellular and parvocellular subpopulations
    # from ipsilateral and contralateral eyes. Pixels correspond roughly to retinal ganglion cells.
    system.add_input(500000, .02, 'Contra-Parvo-RGC') # parvo input should be 3-channel (RGB)
    system.add_input(500000, .02, 'Ipsi-Parvo-RGC')
    system.add_input(500000, .04, 'Contra-Magno-RGC') # magno input should be n-channel luminance video
    system.add_input(500000, .04, 'Ipsi-Magno-RGC')
    system.add('Contra-Parvo-LGN', 0, 0, .02) #TODO: n, e
    system.add('Ipsi-Parvo-LGN', 0, 0, .02)
    system.add('Contra-Magno-LGN', 0, 0, .02)
    system.add('Ipsi-Magno-LGN', 0, 0, .02)

    system.connect_layers('Contra-Parvo-RGC', 'Contra-Parvo-LGN', None)
    system.connect_layers('Ipsi-Parvo-RGC', 'Ipsi-Parvo-LGN', None)
    system.connect_layers('Contra-Magno-RGC', 'Contra-Magno-LGN', None)
    system.connect_layers('Ipsi-Magno-RGC', 'Ipsi-Magno-LGN', None)

    V1_layers = ['2/3', '3B', '4A', '4B', '4Calpha', '4Cbeta', '5', '6'] # we omit the few cells with bodies in layer 1
    layer_map = {'1': '1', '2/3': '2/3', '3B': '2/3', '4': '4', # for looking up extrinsic connections
                 '4A': '4', '4B': '4', '4Calpha': '4', '4Cbeta': '4', '5': '5', '6': '6'}

    for layer in V1_layers:
        n = get_num_neurons('V1', layer)
        e = synapses_per_neuron('V1', 'extrinsic', layer_map[layer])
        system.add(layer, n, e, .1) #TODO: better RF sizes

    system.add('V1_blobs', 0, 0, 0) #TODO: params (in layer 2/3)

    #TODO: consider adding position property to population, None if to be filled in from Yerkes19
    V2_parts = ['Thin-Stripe', 'ThickS-tripe', 'Inter-Stripe']
    V2_layers = ['2/3', '4', '5', '6']
    for part in V2_parts:
        for layer in V2_layers:
            name = '{}-{}'.format(part, layer)
            n = None #TODO
            e = None
            w = None
            system.add(name, n, e, w)




    #TODO: MT should get input from V2 thick stripes and V1 4B
    #TODO: V4 should get input from thin and interstripes and V1 2/3 blobs and interblobs
    #TODO: probably some crosstalk too






def _pop_name(area, layer):
    return '{}_{}'.format(area, layer)

def _add_intrinsic_forward_connections(system, area):
    if area == 'V1':
        # origin, termination, b
        # TODO: arg for dorsal, ventral, or both; or prune areas that don't lead to an output
        # system.connect_layers(_pop_name(area, '4Calpha'), _pop_name(area, '4B'), synapses_per_neuron(area, '4', '2/3'))
        # system.connect_layers(_pop_name(area, '4Cbeta'), _pop_name(area, '3B'), synapses_per_neuron(area, '4', '2/3'))
        # system.connect_layers(_pop_name(area, '4Calpha'), _pop_name(area, '2/3'), synapses_per_neuron(area, '4', '2/3'))
        system.connect_layers(_pop_name(area, '4Cbeta'), _pop_name(area, '2/3'), synapses_per_neuron(area, '4', '2/3'))
        system.connect_layers(_pop_name(area, '2/3'), _pop_name(area, '5'), synapses_per_neuron(area, '2/3', '5'))
        # system.connect_layers(_pop_name(area, '5'), _pop_name(area, '6'), synapses_per_neuron(area, '5', '6'))
    else:
        system.connect_layers(_pop_name(area, '4'), _pop_name(area, '2/3'), synapses_per_neuron(area, '4', '2/3'))
        system.connect_layers(_pop_name(area, '2/3'), _pop_name(area, '5'), synapses_per_neuron(area, '2/3', '5'))
        # system.connect_layers(_pop_name(area, '5'), _pop_name(area, '6'), synapses_per_neuron(area, '5', '6'))


if __name__ == '__main__':
    cortical_areas = ['V1', 'V2', 'V4', 'TEO', 'TEpd']
    # cortical_areas = ['V1', 'V2', 'V3', 'V3A', 'V4', 'MT', 'MST', 'VIP', 'LIP', 'TEO', 'TEpd']
    # cortical_areas = ['V1', 'V2', 'V4', 'TEO', 'TEpd']
    system = make_system(cortical_areas)
    system.prune_FLNe()
    system.check_connected()


    # graph = system.make_graph()
    # import networkx as nx
    # import matplotlib.pyplot as plt
    # from calc.system import get_layout
    # nx.draw_networkx(graph, pos=get_layout(system), arrows=True, font_size=10, node_size=1200, node_color='white')
    # plt.show()

    # for population in system.populations:
    #     if population.name != system.input_name:
    #         population.n = population.n / 10

    calc.conversion.test_stride_patterns(system)

    # from calc.system import get_example_small
    # from calc.conversion import make_net_from_system
    # system = get_example_small()
    # net = make_net_from_system(system)
    # net.make_graph(input_layer, [])


