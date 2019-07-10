# Example systems


import pickle
from calc.system import System
from calc.data import Data, CoCoMac
from calc.optimization import optimize_network_architecture

data = Data()
cocomac = CoCoMac()


def add_areas(system, cortical_areas):
    for area in cortical_areas:
        layers = data.get_layers(area)
        for layer in layers:
            if layer != '1':
                name = _pop_name(area, layer)

                n = _get_num_ff_neurons(area, layer)
                e = data.get_extrinsic_inputs(area, layer) if layer == '4' else None
                w = data.get_receptive_field_size(area) if layer == '4' else None

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

                fl5, fl6 = _get_layer_56_source_fractions(source, target)
                # print('{}->{} fractions L5: {} L6: {}'.format(source, target, fl5, fl6))

                if source == 'V1':
                    if is_ventral(target):
                        # V4 gets 2/3 input, not sure about others
                        system.connect_areas('V1_2/3blob', target_pop, .333*FLNe*SLN/100)
                        system.connect_areas('V1_2/3interblob', target_pop, .667*FLNe*SLN/100)
                        system.connect_areas('V1_5', target_pop, FLNe*(1-SLN/100))
                    else:
                        # true for MT, not sure about others
                        system.connect_areas('V1_4B', target_pop, FLNe*SLN/100)
                elif source == 'V2':
                    if is_ventral(target):
                        thin_fraction = .333
                        pale_fraction = .667
                        system.connect_areas('V2thin_2/3', target_pop, thin_fraction*FLNe*SLN/100)
                        system.connect_areas('V2thin_5', target_pop, fl5*thin_fraction*FLNe*(1-SLN/100))
                        system.connect_areas('V2thin_6', target_pop, fl6*thin_fraction*FLNe*(1-SLN/100))
                        system.connect_areas('V2pale_2/3', target_pop, pale_fraction*FLNe*SLN/100)
                        system.connect_areas('V2pale_5', target_pop, fl5*pale_fraction*FLNe*(1-SLN/100))
                        system.connect_areas('V2pale_6', target_pop, fl6*pale_fraction*FLNe*(1-SLN/100))
                    else:
                        system.connect_areas('V2thick_2/3', target_pop, FLNe*SLN/100)
                        system.connect_areas('V2thick_5', target_pop, fl5*FLNe*(1-SLN/100))
                        system.connect_areas('V2thick_6', target_pop, fl6*FLNe*(1-SLN/100))
                else:
                    system.connect_areas('{}_2/3'.format(source), target_pop, FLNe*SLN/100)
                    system.connect_areas('{}_5'.format(source), target_pop, fl5*FLNe*(1-SLN/100))
                    system.connect_areas('{}_6'.format(source), target_pop, fl6*FLNe*(1-SLN/100))


def _get_layer_56_source_fractions(source_area, target_area):
    result = (0.5, 0.5)

    details = cocomac.get_connection_details(source_area, target_area, guess_missing=True, guess_x=True)

    if details and details['source_layers']:
        l5_strength = float(details['source_layers'][4])
        l6_strength = float(details['source_layers'][5])
        total_strength = l5_strength + l6_strength
        if total_strength > 0:
            result = (l5_strength/total_strength, l6_strength/total_strength)

    return result

def is_ventral(area):
    # see Schmidt et al. (2018) Fig 7 re. V4t
    return area in ['V4', 'VOT', 'PITd', 'PITv', 'CITd', 'CITv', 'AITd', 'AITv', 'TF', 'TH']


def make_big_system(cortical_areas=None):
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
    system.add('magno_LGN', n_magno_LGN, 5, 1.155*w_rf_0)  #TODO reconsider; see Livingston & Hubel (1988)
    system.add('konio_LGN', n_konio_LGN, 5, 1.155*w_rf_0)  #RF sizes highly scattered but comparable to Magno (Xu et al., 2004, J Physiol)
    system.connect_areas(system.input_name, 'parvo_LGN', 1.)
    system.connect_areas(system.input_name, 'magno_LGN', 1.)
    system.connect_areas(system.input_name, 'konio_LGN', 1.)

    # L4A is omitted in classical models, so we leave it out, but see (Sincich et al. 2010)
    for layer in ['4Calpha', '4Cbeta', '4B', '2/3blob', '2/3interblob', '5', '6']:
        if '2/3' in layer:
            n = _get_num_ff_neurons('V1', '2/3')
            if 'blob' in layer:
                n = n / 3  # Sincich et al. (2010, J Neurosci)
            elif 'interblob' in layer:
                n = 2 * n / 3
        else:
            n = _get_num_ff_neurons('V1', layer)

        e = data.get_extrinsic_inputs('V1', '4') if layer.startswith('4C') else None

        # Livingston & Hubel (1988) cite Livingston & Hubel (1984) re larger RF sizes in blobs
        # than interblobs, but I can't find anything about this in the 1984 paper (or elsewhere).
        if '4C' in layer:
            w = data.get_receptive_field_size('V1')
        else:
            w = None
        system.add('V1_{}'.format(layer), n, e, w)

    system.connect_areas('parvo_LGN', 'V1_4Cbeta', 1.)
    system.connect_areas('magno_LGN', 'V1_4Calpha', 1.)
    system.connect_areas('konio_LGN', 'V1_2/3blob', 1.)
    system.connect_areas('konio_LGN', 'V1_2/3interblob', 1.)
    system.connect_layers('V1_4Calpha', 'V1_4B', data.get_inputs_per_neuron('V1', '4', '2/3'))
    system.connect_layers('V1_4Cbeta', 'V1_2/3blob', .5*data.get_inputs_per_neuron('V1', '4', '2/3'))
    system.connect_layers('V1_4Cbeta', 'V1_2/3interblob', data.get_inputs_per_neuron('V1', '4', '2/3'))

    # feedforward magno input to ventral areas (see Merigan & Maunsell, 1983, pg 386)
    system.connect_layers('V1_4Calpha', 'V1_2/3blob', .5*data.get_inputs_per_neuron('V1', '4', '2/3'))

    # Tootell et al. (1988) (IV)
    system.connect_layers('V1_2/3blob', 'V1_5', .5*data.get_inputs_per_neuron('V1', '2/3', '5'))
    system.connect_layers('V1_2/3interblob', 'V1_5', .5*data.get_inputs_per_neuron('V1', '2/3', '5'))
    system.connect_layers('V1_4Cbeta', 'V1_5', data.get_inputs_per_neuron('V1', '4', '5'))

    system.connect_layers('V1_2/3blob', 'V1_6', data.get_inputs_per_neuron('V1', '2/3', '6'))
    system.connect_layers('V1_4Calpha', 'V1_6', data.get_inputs_per_neuron('V1', '4', '6'))
    system.connect_layers('V1_5', 'V1_6', data.get_inputs_per_neuron('V1', '5', '6'))

    for area in ['V2thick', 'V2thin', 'V2pale']:
        for layer in ['2/3', '4', '5', '6']:
            name = _pop_name(area, layer)

            n = _get_num_ff_neurons('V2', layer)
            n = n/5 if 'thin' in area else 2*n/5

            e = data.get_extrinsic_inputs('V2', '4') if layer == '4' else None
            w = data.get_receptive_field_size('V2') if layer == '4' else None

            system.add(name, n, e, w)

        system.connect_layers(_pop_name(area, '4'), _pop_name(area, '2/3'), data.get_inputs_per_neuron('V2', '4', '2/3'))
        system.connect_layers(_pop_name(area, '2/3'), _pop_name(area, '5'), data.get_inputs_per_neuron('V2', '2/3', '5'))
        system.connect_layers(_pop_name(area, '4'), _pop_name(area, '5'), data.get_inputs_per_neuron('V2', '4', '5'))
        system.connect_layers(_pop_name(area, '5'), _pop_name(area, '6'), data.get_inputs_per_neuron('V2', '5', '6'))
        system.connect_layers(_pop_name(area, '2/3'), _pop_name(area, '6'), data.get_inputs_per_neuron('V2', '2/3', '6'))

    # There is substantial forward projection from V1_5 to V2 and V3, also from V2_5 to V3. We assume V1_5 is
    # parvo, and V2_5 is separated into stripes. Will omit parvo, V1_5, and V2thin_5 projections to dorsal areas,
    # consistent with MT supression following magno inactivation, reported in:
    # Maunsell, J. H., Nealey, T. A., & DePriest, D. D. (1990). Magnocellular and parvocellular contributions to
    # responses in the middle temporal visual area (MT) of the macaque monkey. Journal of Neuroscience, 10(10), 3323-3334.
    FLNe = data.get_FLNe('V1', 'V2')
    SLN = data.get_SLN('V1', 'V2')
    system.connect_areas('V1_2/3blob', 'V2thin_4', FLNe * SLN / 100)
    system.connect_areas('V1_2/3interblob', 'V2pale_4', FLNe * SLN / 100)
    system.connect_areas('V1_5', 'V2thin_4', FLNe * (1 - SLN / 100))
    system.connect_areas('V1_4B', 'V2thick_4', FLNe * SLN / 100)

    if system.find_population_index('MT_4'):
        FLNe = data.get_FLNe('V1', 'MT')
        SLN = data.get_SLN('V1', 'MT')
        system.connect_areas('V1_6', 'MT_4', FLNe * (1 - SLN / 100))

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

    if cortical_areas is None:
        cortical_areas = data.get_areas()[:]
        cortical_areas.remove('MDP') # MDP has no inputs in CoCoMac

    add_areas(system, [a for a in cortical_areas if a not in ('V1', 'V2')])
    connect_areas_in_streams(system, cortical_areas)

    # It's correct to have this after making connections, as it operates
    # on actual connections between populations rather than raw FLNe data.
    system.normalize_FLNe()

    system.check_connected()

    return system


def _pop_name(area, layer):
    return '{}_{}'.format(area, layer)


def _get_num_ff_neurons(area, layer):
    n = data.get_num_neurons(area, layer)

    # For L5 and L6, we only want to include cells that contribute to feedforward
    # corticocortical connections, whereas for L2/3 and L4, although many pyramidal cells
    # don't project out of the area, we include them as we assume they have feedforward
    # projections to other layers.

    if layer == '5':
        # In Callaway & Wiser (1996), only 3 of 16 L5 cells project to white matter.
        # Lur et al. (2016) show L5 projecting neurons can be corticocortical, corticotectal,
        # or corticostriatal. They don't give fractions, but we will assume even split and
        # only include corticocortical ones. (3/16)*(1/3) may still be an overestimate as some
        # corticocortical connections are not feedforward.
        return n/16
    elif layer == '6':
        # In L6 we include only group IIA pyramids. These are the only type II cells that enter
        # white matter. Some type I cells enter white matter, but these project to thalamus
        # (Briggs & Callaway, 2001; Briggs, 2010). Wiser & Callaway (1996) say 28% of L6 pyramids
        # project to white matter, a bit less than half of these are short (thalamus-projecting).
        return n*0.15
    else:
        return n


def _add_intrinsic_forward_connections(system, area):
    if area == 'V1':
        system.connect_layers('V1_4Cbeta', 'V1_2/3', data.get_inputs_per_neuron(area, '4', '2/3'))
        system.connect_layers('V1_2/3', 'V1_5', data.get_inputs_per_neuron(area, '2/3', '5'))
        system.connect_layers('V1_4Cbeta', 'V1_5', data.get_inputs_per_neuron('V1', '4', '5'))
    else:
        system.connect_layers(_pop_name(area, '4'), _pop_name(area, '2/3'), data.get_inputs_per_neuron(area, '4', '2/3'))
        system.connect_layers(_pop_name(area, '2/3'), _pop_name(area, '5'), data.get_inputs_per_neuron(area, '2/3', '5'))
        system.connect_layers(_pop_name(area, '4'), _pop_name(area, '5'), data.get_inputs_per_neuron(area, '4', '5'))
        system.connect_layers(_pop_name(area, '2/3'), _pop_name(area, '6'), data.get_inputs_per_neuron(area, '2/3', '6'))
        system.connect_layers(_pop_name(area, '5'), _pop_name(area, '6'), data.get_inputs_per_neuron(area, '5', '6'))


def miniaturize(system, factor=10):
    for population in system.populations:
        if population.name != system.input_name:
            population.n = population.n / factor


if __name__ == '__main__':
    from calc.stride import StridePattern

    # stride_file = 'stride-pattern-PITd.pkl'
    # result_file = 'optimization-result-PITd.pkl'
    # stride_file = 'stride-pattern-compact-PITv.pkl'
    # result_file = 'optimization-result-PITv.pkl'
    stride_file = 'stride-pattern-compact-AITv.pkl'
    result_file = 'optimization-result-AITv.pkl'
    # stride_file = 'stride-pattern-msh.pkl'
    # result_file = 'optimization-result-msh.pkl'

    with open(stride_file, 'rb') as file:
        data = pickle.load(file)

    data['system'].print_description()

    net, training_curve = optimize_network_architecture(data['system'], data['strides'])
    with open(result_file, 'wb') as file:
        pickle.dump({'net': net, 'training_curve': training_curve}, file)

