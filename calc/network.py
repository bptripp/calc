# TODO: consider changing w -> w_k and width -> w_map
import networkx as nx
import numpy as np

class Layer:
    def __init__(self, name, m, width):
        """
        :param name:
        :param m: number of feature maps
        :param width: layer width in pixels
        """
        self.name = name
        self.m = m
        self.width = width


class Connection:
    def __init__(self, pre, post, c, s, w, sigma):
        """
        :param pre: input layer
        :param post: output layer
        :param c: fraction of presynaptic feature maps that contribute to connection
        :param s: stride
        :param w: width of square kernel
        :param sigma: pixel-wise connection sparsity
        """
        self.pre = pre
        self.post = post
        self.c = c
        self.s = s
        self.w = w
        self.sigma = sigma

    def get_name(self):
        return '{}->{}'.format(self.pre.name, self.post.name)


class Network:
    def __init__(self):
        self.layers = []
        self.connections = []

    def add(self, name, m, width):
        result = Layer(name, m, width)
        self.layers.append(result)
        return result

    def connect(self, pre, post, c, s, w, sigma):
        if isinstance(pre, str):
            pre = self.find_layer(pre)
        if isinstance(post, str):
            post = self.find_layer(post)
        result = Connection(pre, post, c, s, w, sigma)
        self.connections.append(result)
        return result

    def find_layer(self, name):
        result = None
        for layer in self.layers:
            if layer.name == name:
                result = layer
                break
        return result

    def find_layer_index(self, name):
        result = None
        for i in range(len(self.layers)):
            if self.layers[i].name == name:
                result = i
                break
        return result

    def find_connection_index(self, pre_name, post_name):
        result = None
        for i in range(len(self.connections)):
            if self.connections[i].pre.name == pre_name and self.connections[i].post.name == post_name:
                result = i
                break
        return result

    def print(self):
        for layer in self.layers:
            print('{} (m={:2.2f} width={:2.2f})'.format(layer.name, layer.m, layer.width))

        for conn in self.connections:
            print('{} -> {} (c={:8.6f} s={:2.2f} w={:2.2f} sigma={:8.6f})'.format(
                conn.pre.name, conn.post.name, conn.c, conn.s, conn.w, conn.sigma))

    def find_inbounds(self, layer_name):
        """
        :param layer_name: Name of a layer in the network
        :return: List of connections that provide input to the layer
        """
        result = []
        for connection in self.connections:
            if connection.post.name == layer_name:
                result.append(connection)
        return result

    def find_outbounds(self, layer_name):
        """
        :param layer_name: Name of a layer in the network
        :return: List of connections out of the layer
        """
        result = []
        for connection in self.connections:
            if connection.pre.name == layer_name:
                result.append(connection)
        return result

    def remove_layer(self, name):
        for connection in self.find_inbounds(name):
            self.remove_connection(connection.pre.name, connection.post.name)

        for connection in self.find_outbounds(name):
            self.remove_connection(connection.pre.name, connection.post.name)

        index = self.find_layer_index(name)
        del self.layers[index]

    def remove_connection(self, pre_name, post_name):
        index = self.find_connection_index(pre_name, post_name)
        del self.connections[index]

    def make_graph(self):
        graph = nx.DiGraph()

        for layer in self.layers:
            graph.add_node(layer.name)

        for connection in self.connections:
            graph.add_edge(connection.pre.name, connection.post.name)

        return graph

    def prune_dead_ends(self, output_layers):
        graph = self.make_graph()

        # keep = []
        remove = []
        for layer in self.layers:
            path_exists_to_output = False
            for output in output_layers:
                if nx.has_path(graph, layer.name, output):
                    path_exists_to_output = True
                    break
            # keep.append(path_exists_to_output)
            if not path_exists_to_output:
                remove.append(layer.name)
                print('Pruning {}'.format(layer.name))

        removed_indices = []
        for layer_name in remove:
            removed_indices.append(self.find_layer_index(layer_name))

        for layer_name in remove:
                self.remove_layer(layer_name)

        return removed_indices

    def scale_c(self, factor):
        """
        Scales all c parameters in log space.
        :param factor factor to scale by
        """
        for connection in self.connections:
            connection.c = np.exp(np.log(connection.c)*factor)

    def scale_sigma(self, factor):
        """
        Scales all sigma parameters in log space.
        :param factor factor to scale by
        """
        for connection in self.connections:
            connection.sigma = np.exp(np.log(connection.sigma)*factor)

