import networkx as nx
import matplotlib.pyplot as plt


class Population:
    def __init__(self, name, n, e, w):
        """
        :param name:
        :param n: # of neurons (estimated from density and cortical area from Markov)
        :param e: # of extrinsic inputs per neuron (typical values by cortical layer)
        :param w: receptive field width in deg visual angle (various sources)
        """
        self.name = name
        self.n = n
        self.e = e
        self.w = w

    def get_description(self):
        return '{} (#neurons={}; in-degree={}; RF-width={})'.format(self.name, self.n, self.e, self.w)

    def is_input(self):
        """
        :return: True if this population is an input to the model (this is true of the # extrinsic inputs
            per neuron is 0); False otherwise
        """

#TODO: do I really need subclasses or can I have f and b, possibly None, f ignored if b not None?
class Projection:
    def __init__(self, origin, termination):
        """
        :param origin: presynaptic Population
        :param termination: postsynaptic Population
        """
        self.origin = origin
        self.termination = termination

    def get_description(self):
        return '{}->{}'.format(self.origin.name, self.termination.name)


class InterAreaProjection(Projection):
    def __init__(self, origin, termination, f):
        """
        :param origin: presynaptic Population
        :param termination: postsynaptic Population
        :param f: fraction of all neurons that project to termination that are from origin (from Markov
            et al., 2012)
        """
        Projection.__init__(self, origin, termination)
        self.f = f

    def get_description(self):
        return '{} (FLNe={})'.format(Projection.get_description(), self.f)


class InterLaminarProjection(Projection):
    def __init__(self, origin, termination, b):
        """
        :param origin: presynaptic Population
        :param termination: postsynaptic Population
        :param b: mean number of synapses onto a single postsynaptic neuron from the presynaptic neuron
            population (b is for Binzegger, since we take this from Binzegger et al., 2004, Figures
            7 and 8)
        """
        Projection.__init__(self, origin, termination)
        self.b = b

    def get_description(self):
        return '{} (synapses-per-target={})'.format(Projection.get_description(self), self.b)


class System:
    def __init__(self):
        self.input_name = 'INPUT'
        self.populations = []
        self.projections = []

    def add_input(self, n, w):
        """
        Adds a special population that represents the network input. If a parameter value is
        unknown, it should be given as None.

        :param n: number of units
        :param w: width of an image pixel in degrees visual angle
        :param name (optional): Defaults to 'INPUT'
        """
        self.populations.append(Population(self.input_name, n, 0, w))

    def add(self, name, n, e, w):
        if self.find_population(name) is not None:
            raise Exception(name + ' already exists in network')

        self.populations.append(Population(name, n, e, w))

    def connect_areas(self, origin_name, termination_name, f):
        origin = self.find_population(origin_name)
        termination = self.find_population(termination_name)

        if origin is None:
            raise Exception(origin_name + ' is not in the system')
        if termination is None:
            raise Exception(termination_name + ' is not in the system')

        self.projections.append(InterAreaProjection(origin, termination, f))

    def connect_layers(self, origin_name, termination_name, b):
        origin = self.find_population(origin_name)
        termination = self.find_population(termination_name)

        if origin is None:
            raise Exception(origin_name + ' is not in the system')
        if termination is None:
            raise Exception(termination_name + ' is not in the system')

        self.projections.append(InterLaminarProjection(origin, termination, b))

    def find_population(self, name):
        assert isinstance(name, str)
        result = None
        for population in self.populations:
            if population.name == name:
                result = population
                break
        return result

    def find_population_index(self, name):
        assert isinstance(name, str)
        result = None
        for i in range(len(self.populations)):
            if self.populations[i].name == name:
                result = i
                break
        return result

    def find_projection(self, origin_name, termination_name):
        assert isinstance(origin_name, str)
        assert isinstance(termination_name, str)
        result = None
        for projection in self.projections:
            if projection.origin.name == origin_name and projection.termination.name == termination_name:
                result = projection
                break
        return result

    def find_projection_index(self, origin_name, termination_name):
        assert isinstance(termination_name, str)
        for i in range(len(self.projections)):
            projection = self.projections[i]
            if projection.origin.name == origin_name and projection.termination.name == termination_name:
                result = i
                break
        return result

    def find_pre(self, termination_name):
        assert isinstance(termination_name, str)

        result = []
        for projection in self.projections:
            if projection.termination.name == termination_name:
                result.append(projection.origin)
        return result

    def prune_FLNe(self):
        """
        The fraction of extrinsic labelled neurons per source area is determined from tract-tracing
        data. However, if a System does not contain all connections in the brain, the sum of these
        fractions will be <1. This method rescales the fractions from the literature to fractions
        within the model.
        """
        for population in self.populations:
            total_FLNe = 0
            for pre in self.find_pre(population.name):
                projection = self.find_projection(pre.name, population.name)
                if isinstance(projection, InterAreaProjection):
                    total_FLNe += projection.f

            for pre in self.find_pre(population.name):
                projection = self.find_projection(pre.name, population.name)
                if isinstance(projection, InterAreaProjection):
                    projection.f = projection.f / total_FLNe

    def make_graph(self):
        graph = nx.DiGraph()

        for population in self.populations:
            graph.add_node(population.name)

        for projection in self.projections:
            graph.add_edge(projection.origin.name, projection.termination.name)

        return graph

    def print_description(self):
        for population in self.populations:
            print(population.get_description())

        for projection in self.projections:
            print(projection.get_description())


def get_example_system():
    result = System()
    result.add_input(250000, .02)
    result.add('V1', 10000000, 2000, .1)
    result.add('V2', 10000000, 2000, .2)
    result.add('V4', 5000000, 2000, .4)
    result.connect_areas('INPUT', 'V1', 1.)
    result.connect_areas('V1', 'V2', 1.)
    result.connect_areas('V1', 'V4', .5)
    result.connect_areas('V2', 'V4', .5)
    return result

def get_example_small():
    result = System()
    result.add_input(750000, .02)
    result.add('V1_4', 53000000, 500, .09)
    result.add('V1_23', 53000000, 1000, .1)
    result.add('V1_5', 27000000, 3000, .11)
    result.add('V2_4', 33000000, 500, .19)
    result.add('V2_23', 33000000, 1000, .2)
    result.add('V2_5', 17000000, 3000, .21)
    result.add('V4_4', 17000000, 500, .39)
    result.add('V4_23', 17000000, 1000, .4)
    result.add('V4_5', 8000000, 3000, .41)

    result.connect_areas('INPUT', 'V1_4', 1.)

    result.connect_layers('V1_4', 'V1_23', 800.)
    result.connect_layers('V1_23', 'V1_5', 3000.)

    result.connect_areas('V1_5', 'V2_4', 1.)

    result.connect_layers('V2_4', 'V2_23', 800.)
    result.connect_layers('V2_23', 'V2_5', 3000.)

    result.connect_areas('V1_5', 'V4_4', .15)
    result.connect_areas('V2_5', 'V4_4', .85)

    result.connect_layers('V4_4', 'V4_23', 800.)
    result.connect_layers('V4_23', 'V4_5', 3000.)

    return result

def get_example_medium():
    # This example was written before the code distinguished interarea and interlaminar
    # connections. Interarea connections are used throughout (even between layers) to
    # preserve it as-is.

    result = System()
    result.add_input(750000, .02)
    result.add('LGN', 2000000, 1000, .04)
    result.add('V1_4', 53000000, 500, .1)
    result.add('V1_23', 53000000, 1000, .13)
    result.add('V2_4', 33000000, 500, .2)
    result.add('V2_23', 33000000, 1000, .26)
    result.add('V4_4', 17000000, 500, .4)
    result.add('V4_23', 17000000, 1000, .5)
    result.add('MT_4', 4800000, 500, 1.)
    result.add('MT_23', 4800000, 1000, 1.1)
    result.add('TEO_4', 6000000, 500, 1.4)
    result.add('TEO_23', 6000000, 1000, 1.5)
    result.add('TEpd_4', 5700000, 500, 3.)
    result.add('TEpd_23', 5700000, 1000, 4.)
    result.add('DP_4', 17000000, 500, 1.7)
    result.add('DP_23', 17000000, 1000, 1.8)

    # input
    result.connect_areas('INPUT', 'LGN', 1.)
    result.connect_areas('LGN', 'V1_4', 1.)

    # laminar connections
    result.connect_areas('V1_4', 'V1_23', 1.)
    result.connect_areas('V2_4', 'V2_23', 1.)
    result.connect_areas('V4_4', 'V4_23', 1.)
    result.connect_areas('MT_4', 'MT_23', 1.)
    result.connect_areas('TEO_4', 'TEO_23', 1.)
    result.connect_areas('TEpd_4', 'TEpd_23', 1.)
    result.connect_areas('DP_4', 'DP_23', 1.)

    # feedforward inter-areal connections
    result.connect_areas('V1_23', 'V2_4', 1.)
    result.connect_areas('V1_23', 'V4_4', 0.0307)
    result.connect_areas('V1_23', 'MT_4', 0.0235)
    result.connect_areas('V2_23', 'V4_4', 0.9693)
    result.connect_areas('V2_23', 'MT_4', 0.2346)
    result.connect_areas('V2_23', 'TEpd_4', 0.0026)
    result.connect_areas('V2_23', 'DP_4', 0.2400)
    result.connect_areas('V4_23', 'MT_4', 0.7419)
    result.connect_areas('V4_23', 'TEpd_4', 0.2393)
    result.connect_areas('V4_23', 'DP_4', 0.7591)
    result.connect_areas('TEO_23', 'TEpd_4', 0.7569)
    result.connect_areas('TEO_23', 'DP_4', 0.0008)
    result.connect_areas('MT_23', 'TEpd_4', 0.0004)
    result.connect_areas('DP_23', 'TEpd_4', 0.0009)
    result.connect_areas('V2_23', 'TEO_4', 0.0909)
    result.connect_areas('V4_23', 'TEO_4', 0.9091)

    return result


def get_layout(sys):
    areas = {}
    areas['INPUT'] = [.075, .5]
    areas['LGN'] = [.17, .5]
    areas['V1'] = [.15, .5]
    areas['V2'] = [.225, .6]
    areas['V3'] = [.3, .7]
    areas['V3A'] = [.35, .7]
    areas['V4'] = [.35, .4]
    areas['TEO'] = [.5, .23]
    areas['MT'] = [.425, .55]
    areas['MST'] = [.5, .55]
    areas['VIP'] = [.7, .7]
    areas['LIP'] = [.65, .65]
    areas['TEpd'] = [.6, .25]
    areas['DP'] = [.4, .5]

    offsets = {}
    offsets['4'] = 0
    offsets['4Cbeta'] = 0
    offsets['23'] = .09
    offsets['2/3'] = .09
    offsets['5'] = -.09

    result = {}
    for pop in sys.populations:
        name = pop.name.split('_')
        position = areas[name[0]].copy()
        if len(name) > 1:
            # print(offsets[name[1]])
            position[1] = position[1] + offsets[name[1]]
        result[pop.name] = position

    print(result)
    return result

if __name__ == '__main__':
    sys = get_example_medium()

    foo = get_layout(sys)
    print(foo)

    graph = sys.make_graph()
    # print(type(nx.drawing.layout.random_layout(graph)))
    #TODO: these layouts all look awful; should use flat-map cortical positions
    nx.draw_networkx(graph, pos=get_layout(sys), arrows=True, font_size=10, node_size=1200, node_color='white')
    # nx.draw_networkx(graph, pos=nx.spring_layout(graph), arrows=True, font_size=10, node_size=1200, node_color='white')
    # nx.draw_networkx(graph, pos=nx.drawing.layout.fruchterman_reingold_layout(graph), arrows=True, font_size=10, node_size=1200, node_color='white')
    plt.show()

    # from calc.conversion import make_net_from_system
    # net = make_net_from_system(sys)
    # net.print()