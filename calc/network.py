# TODO: consider changing w -> w_k and width -> w_map
# TODO: update Network from TF

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


class Network:
    def __init__(self):
        self.layers = []
        self.connections = []

    def add(self, name, m, width):
        self.layers.append(Layer(name, m, width))

    def connect(self, pre, post, c, s, w, sigma):
        self.connections.append(Connection(pre, post, c, s, w, sigma))

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


    def print(self):
        for layer in self.layers:
            print('{} (m={:2.2f} width={:2.2f})'.format(layer.name, layer.m, layer.width))

        for conn in self.connections:
            print('{} -> {} (c={:8.6f} s={:2.2f} w={:2.2f} sigma={:8.6f})'.format(
                conn.pre.name, conn.post.name, conn.c, conn.s, conn.w, conn.sigma))
