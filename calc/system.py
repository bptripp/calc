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


class Projection:
    def __init__(self, origin, termination, f):
        """
        :param origin: presynaptic Population
        :param termination: postsynaptic Population
        :param f: fraction of all neurons that project to termination that are from origin (from Markov)
        """
        self.origin = origin
        self.termination = termination
        self.f = f


class System:
    def __init__(self):
        self.input_name = 'INPUT'
        self.populations = []
        self.projections = []

    def add_input(self, n, w):
        """
        Adds a special population that represents the network input.

        :param n: number of units
        :param w: width of an image pixel in degrees visual angle
        """
        self.populations.append(Population(self.input_name, n, 0, w))

    def add(self, name, n, e, w):
        if self.find_population(name) is not None:
            raise Exception(name + ' already exists in network')

        self.populations.append(Population(name, n, e, w))

    def connect(self, origin_name, termination_name, f):
        origin = self.find_population(origin_name)
        termination = self.find_population(termination_name)

        if origin is None:
            raise Exception(origin_name + ' is not in the system')
        if termination is None:
            raise Exception(termination_name + ' is not in the system')

        self.projections.append(Projection(origin, termination, f))

    def find_population(self, name):
        result = None
        for population in self.populations:
            if population.name == name:
                result = population
                break
        return result

    def find_population_index(self, name):
        result = None
        for i in range(len(self.populations)):
            if self.populations[i].name == name:
                result = i
                break
        return result

    def find_projection(self, origin_name, termination_name):
        result = None
        for projection in self.projections:
            if projection.origin.name == origin_name and projection.termination.name == termination_name:
                result = projection
                break
        return result

    def find_projection_index(self, origin_name, termination_name):
        for i in range(len(self.projections)):
            projection = self.projections[i]
            if projection.origin.name == origin_name and projection.termination.name == termination_name:
                result = i
                break
        return result


    def find_pre(self, termination_name):
        result = []
        for projection in self.projections:
            if projection.termination.name == termination_name:
                result.append(projection.origin)
        return result

    def make_graph(self):
        graph = nx.DiGraph()

        for population in self.populations:
            graph.add_node(population.name)

        for projection in self.projections:
            graph.add_edge(projection.origin.name, projection.termination.name)

        return graph


def get_example_system():
    result = System()
    result.add_input(250000, .02)
    result.add('V1', 10000000, 2000, .1)
    result.add('V2', 10000000, 2000, .2)
    result.add('V4', 5000000, 2000, .4)
    result.connect('INPUT', 'V1', 1.)
    result.connect('V1', 'V2', 1.)
    result.connect('V1', 'V4', .5)
    result.connect('V2', 'V4', .5)
    return result

def get_example_small():
    result = System()
    result.add_input(750000, .02)
    result.add('V1_4', 53000000, 500, .1)
    result.add('V1_23', 53000000, 1000, .1)
    result.add('V1_5', 27000000, 3000, .1)
    result.add('V2_4', 33000000, 500, .2)
    result.add('V2_23', 33000000, 1000, .2)
    result.add('V2_5', 17000000, 3000, .2)
    result.add('V4_4', 17000000, 500, .4)
    result.add('V4_23', 17000000, 1000, .4)
    result.add('V4_5', 8000000, 3000, .4)
    result.connect('INPUT', 'V1_4', 1.)

    result.connect('V1_4', 'V1_23', 1.)
    result.connect('V1_23', 'V1_5', 1.)

    result.connect('V1_5', 'V2_4', 1.)

    result.connect('V2_4', 'V2_23', 1.)
    result.connect('V2_23', 'V2_5', 1.)

    result.connect('V1_5', 'V4_4', .15)
    result.connect('V2_5', 'V4_4', .85)

    result.connect('V4_4', 'V4_23', 1.)
    result.connect('V4_23', 'V4_5', 1.)

    return result

def get_example_medium():
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
    result.connect('INPUT', 'LGN', 1.)
    result.connect('LGN', 'V1_4', 1.)

    # laminar connections
    result.connect('V1_4', 'V1_23', 1.)
    result.connect('V2_4', 'V2_23', 1.)
    result.connect('V4_4', 'V4_23', 1.)
    result.connect('MT_4', 'MT_23', 1.)
    result.connect('TEO_4', 'TEO_23', 1.)
    result.connect('TEpd_4', 'TEpd_23', 1.)
    result.connect('DP_4', 'DP_23', 1.)

    # feedforward inter-areal connections
    result.connect('V1_23', 'V2_4', 1.)
    result.connect('V1_23', 'V4_4', 0.0307)
    result.connect('V1_23', 'MT_4', 0.0235)
    result.connect('V2_23', 'V4_4', 0.9693)
    result.connect('V2_23', 'MT_4', 0.2346)
    result.connect('V2_23', 'TEpd_4', 0.0026)
    result.connect('V2_23', 'DP_4', 0.2400)
    result.connect('V4_23', 'MT_4', 0.7419)
    result.connect('V4_23', 'TEpd_4', 0.2393)
    result.connect('V4_23', 'DP_4', 0.7591)
    result.connect('TEO_23', 'TEpd_4', 0.7569)
    result.connect('TEO_23', 'DP_4', 0.0008)
    result.connect('MT_23', 'TEpd_4', 0.0004)
    result.connect('DP_23', 'TEpd_4', 0.0009)
    result.connect('V2_23', 'TEO_4', 0.0909)
    result.connect('V4_23', 'TEO_4', 0.9091)

    return result


def get_layout(sys):
    areas = {}
    areas['INPUT'] = [.15, .5]
    areas['V1'] = [.2, .5]
    areas['V2'] = [.25, .6]
    areas['V4'] = [.35, .4]
    areas['TEO'] = [.4, .3]
    areas['MT'] = [.45, .45]
    areas['TEpd'] = [.5, .25]
    areas['DP'] = [.4, .6]

    offsets = {}
    offsets['4'] = 0
    offsets['23'] = .09
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