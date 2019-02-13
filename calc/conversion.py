from calc.system import InterLaminarProjection, InterAreaProjection
from calc.network import Network
import tensorflow as tf
import numpy as np

#TODO: refactor finer-grained cost objects that encapsulate individual terms, associated NetworkVariables & SystemConstants


class SystemConstants:
    """
    TensorFlow constants for parameters of a System.
    """

    def __init__(self, system):
        """
        :param system: physiological network model to approximate with convolutional network architecture
        """

        self.n = []
        self.e = []
        self.w = []
        self.f = []
        self.b = []

        for population in system.populations:
            self.n.append(tf.constant(float(population.n), name='n_in_{}'.format(population.name)))

            if population.w:
                self.w.append(tf.constant(float(population.w), name='w_of_{}'.format(population.name)))
            else:
                self.w.append(None)

            if population.name == system.input_name or population.e is None:
                self.e.append(None)
            else:
                self.e.append(tf.constant(float(population.e), name='e_of_{}'.format(population.name)))

        for projection in system.projections:
            if isinstance(projection, InterAreaProjection):
                # print('{} f={}'.format(projection.get_description(), projection.f))
                self.f.append(tf.constant(float(projection.f)))
                self.b.append(None)
            elif isinstance(projection, InterLaminarProjection):
                self.f.append(None)
                self.b.append(tf.constant(float(projection.b)))


def get_variable(name, initial_value):
    return tf.get_variable(name, initializer=tf.constant(float(initial_value)))


class NetworkVariables:
    """
    TensorFlow variables for parameters of a Network.
    """

    def __init__(self, network, system, image_layer, image_pixel_width):
        """
        :param network: model of convnet architecture
        :param image_layer: index of the image layer (the input)
        :param image_pixel_width: width of an image pixel in degrees visual angle
        """
        self.image_layer = image_layer
        self.image_pixel_width = image_pixel_width

        self.n_layers = len(network.layers)
        self.n_connections = len(network.connections)

        # hyperparameters wrapped as TF variables
        self.c = [] # fraction of feature maps in pre layer that contribute to each connection
        self.sigma = [] # pixel-wise sparsity of each connection

        # an additional variable from which some connection hyperparams are derived
        self.w_rf = [] # RF width of each layer in degrees visual angle

        # set separately (integer parameters not suitable for gradient descent)
        self.s = [] # stride of each connection
        self.m = [] # number of feature maps in each layer
        self.width = [] # width of feature maps each layer

        # derived from w_rf and other hyperparams
        self.w = [] # kernel widths

        # supporting information ...
        self.input_layers = [] #for each layer, a list of indices of layers that provide its input
        self.input_connections = [] # for each layer, a list of indices of connections that provide input
        self.output_connections = [] # for each layer, a list of indices of connections that carry output
        self.pres = [] # for each connection, index of presynaptic layer
        self.posts = [] # for each connection, index of postsynaptic layer
        self.inter_area = [] # for each connection, True for InterAreaProjections
        self.cortical_layer = [] # for each layer, cortical layer name or None (used for dead-end cost)

        for layer in network.layers:
            self.m.append(tf.constant(layer.m, dtype=tf.float32))
            self.width.append(tf.constant(layer.width, dtype=tf.float32))

            input_layers = []
            input_connections = []
            output_connections = []
            for i in range(len(network.connections)):
                connection = network.connections[i]
                if connection.post.name == layer.name:
                    input_layers.append(network.find_layer_index(connection.pre.name))
                    input_connections.append(i)
                if connection.pre.name == layer.name:
                    output_connections.append(i)
            self.input_layers.append(input_layers)
            self.input_connections.append(input_connections)
            self.output_connections.append(output_connections)

            cortical_layer = None
            if '_' in layer.name:
                cortical_layer = layer.name.split('_')[1]
            self.cortical_layer.append(cortical_layer)

        w_rf_init = self._min_w_rfs(system, network)
        for i in range(len(w_rf_init)):
            self.w_rf.append(get_variable('{}-w_rf_s'.format(network.layers[i].name), w_rf_init[i]))

        for connection in network.connections:
            conn_name = '{}-{}'.format(connection.pre.name, connection.post.name)

            pre_ind = network.find_layer_index(connection.pre.name)
            post_ind = network.find_layer_index(connection.post.name)

            self.c.append(get_variable('{}-c'.format(conn_name), connection.c))
            self.sigma.append(get_variable('{}-s'.format(conn_name), connection.sigma))

            self.s.append(tf.constant(connection.s, dtype=tf.float32))

            pre_pixel_width = image_pixel_width * self.width[image_layer] / self.width[pre_ind]

            # convert RF sizes from degrees visual angle to *pre-layer* pixels (units of the kernel)...
            sigma_post = tf.divide(self.w_rf[post_ind], pre_pixel_width)
            sigma_pre = tf.divide(self.w_rf[pre_ind], pre_pixel_width)
            sigma_kernel = tf.sqrt(sigma_post**2 - sigma_pre**2)
            w = tf.constant(12**.5) * sigma_kernel  
            self.w.append(w)

            self.pres.append(pre_ind)
            self.posts.append(post_ind)

        for projection in system.projections:
            self.inter_area.append(isinstance(projection, InterAreaProjection))

    def _min_w_rfs(self, system, network):
        w_rf_init = [pop.w for pop in system.populations]
        done = False
        while not done:
            done = True
            for i in range(len(w_rf_init)):
                if w_rf_init[i] is None:
                    min_downstream_rfs = []
                    for input_layer in self.input_layers[i]: # layers that provide direct input to this one
                        w_rf_input = w_rf_init[input_layer]
                        width_input = network.layers[input_layer].width
                        if w_rf_input is None:
                            break
                        else:
                            min_downstream_rfs.append(
                                self._get_min_downstream_w_rf(w_rf_input,
                                                              width_input,
                                                              network.layers[self.image_layer].width)
                            )
                    if len(min_downstream_rfs) == len(self.input_layers[i]):
                        w_rf_init[i] = np.max(min_downstream_rfs)
                    else:
                        done = False

        return w_rf_init

    def _get_min_downstream_w_rf(self, w_rf, layer_width, image_layer_width):
        sigma = w_rf / self.image_pixel_width * layer_width / image_layer_width
        sigma_downstream = np.sqrt(sigma**2 + 1/12)
        return w_rf * sigma_downstream / sigma


def get_clip_ops(vars, min=1e-4, max=1.):
    # ops for clipping all variables to > 0 and <= 1

    result = []
    for var in vars:
        op = tf.assign(var, tf.clip_by_value(var, min, max))
        result.append(op)

    return result


def norm_squared_error(target, actual):
    return tf.square(tf.log(tf.divide(actual, target)))


class Cost:
    def __init__(self, system, network, image_layer=0):
        """
        :param system: A system.System (physiological system definition)
        :param network: A network.Network (convnet definition)
        :param image_layer: index of the input-image layer (defaults to 1)
        """
        self.system = SystemConstants(system)
        self.network = NetworkVariables(network, system, image_layer, system.populations[image_layer].w)

    def _get_n_network(self, i):
        return self.network.m[i] * tf.square(self.network.width[i])

    def match_cost_n(self, kappa):
        """
        :param kappa: weight relative to other costs
        :return: TF node for the total cost of population/layer size mismatches
        """
        terms = []
        for i in range(len(self.system.n)):
            n_system = self.system.n[i]
            n_network = self._get_n_network(i)
            terms.append(norm_squared_error(n_system, n_network))

        return tf.multiply(tf.constant(kappa), tf.reduce_mean(terms))

    def match_cost_e(self, kappa):
        """
        :param kappa: weight relative to other costs
        :return: TF node for the total cost of mismatches in number of extrinsic inputs per neuron
        """
        terms = []
        for i in range(len(self.system.e)):
            if self.system.e[i] is not None: #skip the input layer, which has no extrinsic inputs
                e_system = self.system.e[i]
                e_network = self._get_e_network(i)

                term = norm_squared_error(e_system, e_network)
                terms.append(term)

        return tf.constant(kappa) * tf.reduce_mean(terms)

    def _get_e_network(self, i):
        input_layers = self.network.input_layers[i]
        input_connections = self.network.input_connections[i]
        subtotals = []
        for i in range(len(input_layers)):
            il = input_layers[i]
            ic = input_connections[i]
            subtotal = tf.square(self.network.w[ic]) * self.network.c[ic] * self.network.sigma[ic] * self.network.m[il]
            subtotals.append(subtotal)
        return tf.reduce_sum(subtotals)

    def match_cost_w(self, kappa):
        """
        :param kappa: weight relative to other costs
        :return: TF node for the total cost of population/layer RF mismatches
        """
        terms = []
        for i in range(len(self.system.w)):
            if self.system.w[i] is not None:  # omit from cost if target is None
                w_rf_system = self.system.w[i]
                w_rf_network = self.network.w_rf[i]
                terms.append(norm_squared_error(w_rf_system, w_rf_network))

        return tf.multiply(tf.constant(kappa), tf.reduce_mean(terms))

    def match_cost_b(self, kappa):
        #TODO: unit test
        """
        :param kappa: weight relative to other costs
        :return: TF node for the total cost of mismatches in # of synapses onto each neuron from
            each projection
        """
        terms = []
        for ij in range(len(self.system.b)):
            b_system = self.system.b[ij]
            if b_system is not None:
                b_network = self._get_b_network(ij)
                terms.append(norm_squared_error(b_system, b_network))

        if len(terms) == 0:
            return tf.constant(0.)
        else:
            return tf.constant(kappa) * tf.reduce_mean(terms)

    def _get_b_network(self, ij):
        j = self.network.pres[ij]

        w_k_ij = self.network.w[ij]
        m_j = self.network.m[j]
        c_ij = self.network.c[ij]
        sigma_ij = self.network.sigma[ij]

        return w_k_ij ** 2 * m_j * c_ij * sigma_ij

    def match_cost_f(self, kappa):
        """
        :param kappa: weight relative to other costs
        :return: TF node for the total cost of mismatches in number of extrinsic inputs per neuron
        """

        terms = []
        for ij in range(len(self.system.f)): #looping through connections
            f_system = self.system.f[ij]
            if f_system is not None:
                f_network = self._get_f_network(ij)
                term = norm_squared_error(f_system, f_network)
                terms.append(term)

        if len(terms) == 0:
            return tf.constant(0.)
        else:
            return tf.constant(kappa) * tf.reduce_mean(terms)

    def _get_f_network(self, ij):
        n_network_ij = self.n_contrib(ij)
        n_network_j = []
        input_connections = self.network.input_connections[self.network.posts[ij]]
        for input_connection in input_connections:
            n_network_j.append(self.n_contrib(input_connection))
        return n_network_ij / tf.reduce_sum(n_network_j)

    def n_contrib(self, conn_ind):
        """
        Note this is used in the calculation of f.

        :param conn_ind: index of a connection
        :return: number of presynaptic neurons that contribute to the connection (a TF Node)
        """
        pre_ind = self.network.pres[conn_ind]

        n_j = self._get_n_network(pre_ind)
        c_ij = self.network.c[conn_ind]
        s_ij = self.network.s[conn_ind]
        w_ij = self.network.w[conn_ind]
        alpha_ij = tf.minimum(tf.constant(1.), tf.square(w_ij / s_ij))
        return n_j * c_ij * self.sigma_star(conn_ind) * alpha_ij

    def sigma_star(self, conn_ind):
        """
        Note this is used in the calculation of f.

        :param conn_ind: index of a connection
        :return: fraction of units in a sparse connection that have a non-zero
            influence on at least one postsynaptic unit (a TF Node)
        """
        post_ind = self.network.posts[conn_ind]

        sigma_ij = self.network.sigma[conn_ind]
        w_ij = self.network.w[conn_ind]
        s_ij = self.network.s[conn_ind]
        m_i = self.network.m[post_ind]

        beta_ij = tf.maximum(tf.constant(1.), tf.square(w_ij / s_ij))
        exponent = beta_ij * m_i
        return tf.constant(1.) - tf.pow(tf.constant(1.)-sigma_ij, exponent)

    def param_cost(self, kappa):
        """
        :param kappa: weight relative to other costs (should be quite small)
        :return: cost due to number of weight parameters in all connections
        """
        terms = []
        for i in range(len(self.network.w)):
            w_ij = self.network.w[i]
            m_i = self.network.m[self.network.posts[i]]
            m_j = self.network.m[self.network.pres[i]]
            terms.append(tf.square(w_ij) * m_i * m_j)
        return tf.constant(kappa) * tf.reduce_mean(terms)

    def dead_end_cost(self, kappa):
        """
        :param kappa: weight relative to other costs
        :return: cost due to under-use or over-use of feature maps in outgoing connections;
            the premise is that most pyramidal neurons are projection neurons that have a
            single cortico-cortical connection; we take this to mean that the sum of c over
            outgoing connections should be about 1
        """
        terms = []
        for i in range(self.network.n_layers):
            interlaminar_fractions = []
            interarea_fractions = []
            if self.network.output_connections[i]: # this cost only applies if layer has outputs
                for conn_ind in self.network.output_connections[i]:
                    if self.network.inter_area[conn_ind]:
                        interarea_fractions.append(self.network.c[conn_ind])
                    else:
                        interlaminar_fractions.append(self.network.c[conn_ind])

                layer = '' if self.network.cortical_layer[i] is None else self.network.cortical_layer[i]

                if '2/3' in layer:
                    # most neurons in L2/3 should project to L5
                    # about half should project into white matter (Callaway & Wiser, 1996, Visual Neuroscience)
                    if interarea_fractions:
                        terms.append(tf.square(tf.reduce_sum(interarea_fractions)-0.5))
                    terms.append(tf.square(tf.reduce_sum(interlaminar_fractions)-1.0))
                elif '4' in layer:
                    terms.append(tf.square(tf.reduce_sum(interlaminar_fractions)-1.0))
                elif '5' in layer:
                    # a minority of L5 and L6 neurons project out, but the other ones aren't
                    # included in the feedforward model
                    if interarea_fractions:
                        terms.append(tf.square(tf.reduce_sum(interarea_fractions)-1.0))
                    terms.append(tf.square(tf.reduce_sum(interlaminar_fractions) - 1.0))
                elif '6' in layer:
                    # a minority of L5 and L6 neurons project out, but the other ones aren't
                    # included in the feedforward model
                    if interarea_fractions:
                        terms.append(tf.square(tf.reduce_sum(interarea_fractions)-1.0))
                else:
                    # assume all are projection cells otherwise (e.g. for LGN)
                    if interarea_fractions:
                        terms.append(tf.square(tf.reduce_sum(interarea_fractions)-1.0))

        return tf.constant(kappa) * tf.reduce_mean(terms)

    def w_k_constraint_cost(self, kappa):
        """
        :param kappa: weight relative to other costs
        :return: Cost for soft constraint w_k >= 1
        """
        return kappa * bounds(self.network.w, min_bound=1.)

    def compare_system(self, system, sess):
        """
        Prints comparison of system target properties with properties calculated from network.
        """

        for i in range(len(system.populations)):
            pop = system.populations[i]
            n = round(sess.run(self._get_n_network(i)))
            e = sess.run(self._get_e_network(i))
            if self.network.w_rf[i] is None:
                w = -1
            else:
                w = sess.run(self.network.w_rf[i])
            print('{}, {}, {}, {:10.6f}, {}, {:10.6f};'.format(pop.n, n, pop.e, e, pop.w, w))

        for ij in range(len(system.projections)):
            projection = system.projections[ij]
            if isinstance(projection, InterAreaProjection):
                f = sess.run(self._get_f_network(ij))
                print('0, {}, {:10.6f};'.format(projection.f, f))
            elif isinstance(projection, InterLaminarProjection):
                b = sess.run(self._get_b_network(ij))
                print('1, {}, {:10.6f};'.format(projection.b, b))


def bounds(var_list, min_bound=None, max_bound=None):
    terms = []
    for i in range(len(var_list)):
        if min_bound is not None:
            terms.append(constraint_gt(var_list[i], min_bound))
        if max_bound is not None:
            terms.append(constraint_lt(var_list[i], max_bound))
    return tf.reduce_sum(terms)


def constraint_gt(var, val):
    """
    :param var: TF variable
    :param val: reference value
    :return: threshold-linear cost for constraining var to be greater than val
    """
    return tf.maximum(tf.constant(val) - var, tf.constant(0.))


def constraint_lt(var, val):
    """
    :param var: TF variable
    :param val: reference value
    :return: threshold-linear cost for constraining var to be less than val
    """
    return tf.maximum(var - tf.constant(val), tf.constant(0.))


#TODO: this is only used in a test; remove test or move to test file
def make_net_from_system(system, image_layer=0, image_channels=3.):
    """
    :return: A neural network architecture with the same nodes and connections as the given
        neurophysiological system architecture, and otherwise random hyperparameters
        (to be optimized separately)
    """
    net = Network()

    for i in range(len(system.populations)):
        pop = system.populations[i]
        units = pop.n

        if i == image_layer:
            channels = image_channels
            pixels = round(np.sqrt(units/image_channels))
        else:
            ratio_channels_over_pixels = np.exp(-1.5 + 3*np.random.rand())
            pixels = round(np.cbrt(units / ratio_channels_over_pixels))
            channels = max(1, round(ratio_channels_over_pixels * pixels))
        net.add(pop.name, channels, pixels)

    # try to set strides to reasonable values
    for i in range(len(system.populations)):
        pres = system.find_pre(system.populations[i].name)
        units = net.layers[i].m * net.layers[i].width**2

        if len(pres) > 0:
            pre_ind = system.find_population_index(pres[0].name)
            net.layers[i].width = net.layers[pre_ind].width / 1.5
            net.layers[i].m = units / net.layers[i].width**2

    for projection in system.projections:
        pre = net.find_layer(projection.origin.name)
        post = net.find_layer(projection.termination.name)

        c = .1 + .2*np.random.rand()
        s = 1. + 9.*np.random.rand()
        rf_ratio = projection.termination.w / projection.origin.w
        w = (rf_ratio - 1.) / (0.5 + np.random.rand())
        w = np.maximum(.1, w)  # make sure kernel has +ve width
        sigma = .1 + .1*np.random.rand()
        net.connect(pre, post, c, s, w, sigma)

    return net


def update_net_from_tf(sess, net, nv):
    m = sess.run(nv.m)
    width = sess.run(nv.width)
    for i in range(len(net.layers)):
        # print('layer {} of {}'.format(i, len(net.layers)))
        net.layers[i].m = m[i]
        net.layers[i].width = width[i]

    c = sess.run(nv.c)
    s = sess.run(nv.s)
    w = sess.run(nv.w)
    sigma = sess.run(nv.sigma)
    for i in range(len(net.connections)):
        # print('connection {} of {}'.format(i, len(net.connections)))
        net.connections[i].c = c[i]
        net.connections[i].s = s[i]
        net.connections[i].w = w[i]
        net.connections[i].sigma = sigma[i]


# def print_gradients(cost, vars):
#     gradients = tf.gradients(cost, vars)
#     for i in range(len(vars)):
#         if gradients[i] is not None:
#             print('grad wrt {}: {}'.format(vars[i].name, sess.run(gradients[i])))

