from calc.system import InterLaminarProjection, InterAreaProjection
from calc.network import Network
import tensorflow as tf
import numpy as np

#TODO: refactor finer-grained cost objects that encapsulate individual terms, associated NetworkVariables & SystemConstants

class SystemConstants:
    def __init__(self, system):
        """
        Creates arrays of TensorFlow constants for parameters of a System.

        :param system:
        """
        self.n = []
        self.e = []
        self.w = []
        self.f = []
        self.b = []

        for population in system.populations:
            self.n.append(tf.constant(float(population.n), name='n_in_{}'.format(population.name)))
            self.w.append(tf.constant(float(population.w), name='w_of_{}'.format(population.name)))
            if population.name == system.input_name:
                self.e.append(None)
            else:
                self.e.append(tf.constant(float(population.e), name='e_of_{}'.format(population.name)))

        for projection in system.projections:
            if isinstance(projection, InterAreaProjection):
                self.f.append(tf.constant(float(projection.f)))
                self.b.append(None)
            elif isinstance(projection, InterLaminarProjection):
                self.f.append(None)
                self.b.append(tf.constant(float(projection.b)))


def get_variable(name, initial_value):
    return tf.get_variable(name, initializer=tf.constant(float(initial_value)))


class NetworkVariables:
    def __init__(self, network, system, image_layer, image_pixel_width):
        """
        Creates arrays of TensorFlow variables for parameters of a Network.

        :param network: TF variables will describe this Network
        :param image_layer: index of the image layer (the input)
        :param image_pixel_width: width of an image pixel in degrees visual angle
        """

        self.image_layer = image_layer
        self.image_pixel_width = image_pixel_width

        self.n_layers = len(network.layers)
        self.n_connections = len(network.connections)

        # hyperparameters wrapped as TF variables ...
        self.m = [] # number of feature maps in each layer
        self.width = [] # width of feature maps each layer
        self.m_scaled = []
        self.width_scaled = []

        self.c = [] # fraction of feature maps in pre layer that contribute to each connection
        self.sigma = [] # pixel-wise sparsity of each connection

        # an additional variable from which some connection hyperparams are derived ...
        self.w_rf = [] # RF width of each layer in degrees visual angle
        self.w_rf_scaled = []

        # hyperparams derived from w_rf and other hyperparams
        self.s = [] # stride of each connection
        self.w = [] # kernel widths

        # supporting information ...
        self.input_layers = [] #for each layer, a list of indices of layers that provide its input
        self.input_connections = [] # for each layer, a list of indices of connections that provide input
        self.pres = [] # for each connection, index of presynaptic layer
        self.posts = [] # for each connection, index of postsynaptic layer

        for layer in network.layers:
            # rescale large variables for optimization
            scale = (layer.m + layer.width) / 2.
            scale_constant = tf.constant(scale, dtype=tf.float32)

            m_scaled = get_variable('{}-m_s'.format(layer.name), layer.m/scale)
            self.m_scaled.append(m_scaled)

            width_scaled = get_variable('{}-width_s'.format(layer.name), layer.width/scale)
            self.width_scaled.append(width_scaled)

            self.m.append(scale_constant * m_scaled)
            self.width.append(scale_constant * width_scaled)

            input_layers = []
            input_connections = []
            for i in range(len(network.connections)):
                connection = network.connections[i]
                if connection.post.name == layer.name:
                    input_layers.append(network.find_layer_index(connection.pre.name))
                    input_connections.append(i)
            self.input_layers.append(input_layers)
            self.input_connections.append(input_connections)

        # TODO: should initialize unknowns as falling between surrounding values
        # set RF width as in system if defined; otherwise randomize
        for population in system.populations:
            w_rf = 2. + 5.*np.random.rand() if population.w is None else population.w

            scale = w_rf
            scale_constant = tf.constant(scale, dtype=tf.float32)

            w_rf_scaled = get_variable('{}-w_rf_s'.format(population.name), 1.)
            self.w_rf_scaled.append(w_rf_scaled)
            self.w_rf.append(scale_constant * w_rf_scaled)

        # replace image-channels variable with constant (it shouldn't change) ...
        self.m[image_layer] = tf.constant(float(network.layers[image_layer].m))
        self.m_scaled[image_layer] = None

        for connection in network.connections:
            conn_name = '{}-{}'.format(connection.pre.name, connection.post.name)

            pre_ind = network.find_layer_index(connection.pre.name)
            post_ind = network.find_layer_index(connection.post.name)

            self.c.append(get_variable('{}-c'.format(conn_name), connection.c))
            self.sigma.append(get_variable('{}-s'.format(conn_name), connection.sigma))

            # derived parameters ...
            stride_name = '{}_{}_stride'.format(connection.pre.name, connection.post.name)
            self.s.append(tf.divide(self.width[pre_ind], self.width[post_ind], name=stride_name)) # width j over width i

            pre_pixel_width = image_pixel_width * self.width[image_layer] / self.width[pre_ind]
            # convert RF sizes from degrees visual angle to pre-layer pixels ...
            #TODO: deal with equal RF sizes
            sigma_post = tf.divide(self.w_rf[post_ind], pre_pixel_width)
            sigma_pre = tf.divide(self.w_rf[pre_ind], pre_pixel_width)
            sigma_kernel = tf.sqrt(sigma_post**2 - sigma_pre**2)
            w = tf.constant(12**.5) * sigma_kernel
            self.w.append(w)

            self.pres.append(pre_ind)
            self.posts.append(post_ind)

            # init = tf.global_variables_initializer()
            # with tf.Session() as sess:
            #     sess.run(init)
            #     print(conn_name)
            #     print('w: {} sigma_kernel: {} sigma_pre: {} sigma_post: {}'.format(sess.run(w), sess.run(sigma_kernel), sess.run(sigma_pre), sess.run(sigma_post)))

    def collect_variables(self):
        vars = []

        # note one of the m's is a constant
        for i in range(len(self.m)):
            if self.m_scaled[i] is not None:
                vars.append(self.m_scaled[i])

        vars.extend(self.width_scaled)
        vars.extend(self.c)
        vars.extend(self.w_rf_scaled)
        vars.extend(self.sigma)

        return vars


def get_clip_ops(vars, min=1e-3, max=1.):
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

                # init = tf.global_variables_initializer()
                # with tf.Session() as sess:
                #     sess.run(init)
                #     print('e: {} {} term: {}'.format(sess.run(e_system), sess.run(e_network), sess.run(term)))

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
        c_ij = self.network.c[conn_ind]
        m_i = self.network.m[post_ind]

        beta_ij = tf.maximum(tf.constant(1.), tf.square(w_ij / s_ij))
        exponent = beta_ij * c_ij * m_i
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
        return tf.constant(kappa) * tf.reduce_sum(terms)

    def constraint_cost(self, kappa):
        """
        :param kappa: weight relative to other costs
        :return: Cost for soft constraints on parameters
        """
        return bounds(self.network.w, min=1.) \
            + bounds(self.network.m, min=1.) \
            + bounds(self.network.width, min=1.) \
            + bounds(self.network.s, min=0.1) \
            + bounds(self.network.c, min=1e-2, max=1.) \
            + bounds(self.network.sigma, min=1e-3, max=1.) \
            + bounds(self.network.w_rf, min=1e-3)

    def compare_system(self, system, sess):
        """
        Prints comparison of system target properties with properties calculated from network.
        """

        # This is a good place to add debugging code

        for i in range(len(system.populations)):
            pop = system.populations[i]
            n = round(sess.run(self._get_n_network(i)))
            e = sess.run(self._get_e_network(i))
            w = sess.run(self.network.w_rf[i])
            print('{} n:[{}|{}] e:[{}|{:10.6f}] w:[{}|{:10.6f}]'.format(pop.name, pop.n, n, pop.e, e, pop.w, w))

        for ij in range(len(system.projections)):
            projection = system.projections[ij]
            if isinstance(projection, InterAreaProjection):
                f = sess.run(self._get_f_network(ij))
                print('{}->{} f:[{}|{:10.6f}]'.format(projection.origin.name, projection.termination.name, projection.f, f))
            elif isinstance(projection, InterLaminarProjection):
                b = sess.run(self._get_b_network(ij))
                print('{}->{} b:[{}|{:10.6f}]'.format(projection.origin.name, projection.termination.name, projection.b, b))


def bounds(var_list, min=None, max=None):
    terms = []
    for i in range(len(var_list)):
        if min is not None:
            terms.append(constraint_gt(var_list[i], min))
        if max is not None:
            terms.append(constraint_lt(var_list[i], max))
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
            channels = round(ratio_channels_over_pixels * pixels)
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

        # c = projection.f
        c = .1 + .2*np.random.rand()
        s = 1. + 9.*np.random.rand()
        rf_ratio = projection.termination.w / projection.origin.w
        w = (rf_ratio - 1.) / (0.5 + np.random.rand())
        w = np.maximum(.1, w)  # make sure kernel has +ve width
        sigma = .1 + .1*np.random.rand()
        net.connect(pre, post, c, s, w, sigma)

    return net


# def adjust_net(system, net):
#     for i in range(len(system.projections)):
#         target_f = system.projections[i].f
#
#         total_dense_inputs = 0
#         termination_name = system.projections[i].termination.name
#         for origin_name in system.find_pre(termination_name):
#             conn_ind = system.find_projection_index(origin_name, termination_name)
#             layer_ind = system.find_population_index(origin_name)
#             w = net.connections[conn_ind].w
#             m = net.layers[layer_ind].m
#             total_dense_inputs = total_dense_inputs + w**2 * m


def update_net_from_tf(sess, net, nv):
    for i in range(len(net.layers)):
        net.layers[i].m = sess.run(nv.m[i])
        net.layers[i].width = sess.run(nv.width[i])

    for i in range(len(net.connections)):
        net.connections[i].c = sess.run(nv.c[i])
        net.connections[i].s = sess.run(nv.s[i])
        net.connections[i].w = sess.run(nv.w[i])
        net.connections[i].sigma = sess.run(nv.sigma[i])


def optimize_net(sess, opt_op, clip_ops=[]):

    # gradients = tf.gradients(c, vars)
    for i in range(100):
        # for i in range(len(vars)):
        #     if gradients[i] is not None:
        #         print('grad wrt {}: {}'.format(vars[i].name, sess.run(gradients[i])))
        opt_op.run()

        for clip_op in clip_ops:
            sess.run(clip_op)


def print_gradients(cost, vars):
    gradients = tf.gradients(cost, vars)
    for i in range(len(vars)):
        if gradients[i] is not None:
            print('grad wrt {}: {}'.format(vars[i].name, sess.run(gradients[i])))


if __name__ == '__main__':
    from calc.system import get_example_small, get_example_medium
    system = get_example_small()
    # system = get_example_medium()

    net = make_net_from_system(system)
    cost = Cost(system, net)
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     update_net_from_tf(sess, net, cost.network)
    # adjust_net(system, net)
    # cost = Cost(system, net)


    # c = cost.match_cost_f(1.)
    c = cost.match_cost_n(1.) \
        + cost.match_cost_w(1.) \
        + cost.match_cost_e(1.) \
        + cost.match_cost_f(1.) \
        + cost.match_cost_b(1.) \
        + cost.constraint_cost(1.)
    # c = cost.match_cost_w(1.) + cost.match_cost_e(1.) + cost.match_cost_f(1.) + cost.constraint_cost(1.)
    # c = cost.match_cost_n(1.) + cost.match_cost_w(1.) + cost.match_cost_e(1.) + cost.constraint_cost(1.)
    # + cost.param_cost(.000001)


    # opt_slow = tf.train.AdamOptimizer(learning_rate=.00001, epsilon=10-4)
    opt_slow = tf.train.AdamOptimizer()
    vars = cost.network.collect_variables()
    opt_op = opt_slow.minimize(c, var_list=vars)

    opt_fast = tf.train.AdamOptimizer()
    c_e = cost.match_cost_e(1.)
    opt_op_e = opt_fast.minimize(c_e, var_list=cost.network.sigma)

    c_partial = cost.match_cost_f(1.) + cost.match_cost_e(1.)
    vars_partial = []
    vars_partial.extend(cost.network.c)
    vars_partial.extend(cost.network.sigma)
    opt_op_partial = opt_fast.minimize(c_partial, var_list=vars_partial)

    clip_vars = []
    clip_vars.extend(cost.network.c)
    clip_vars.extend(cost.network.sigma)

    clip_ops = get_clip_ops(clip_vars)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        # cost.compare_system(system, sess)

        # update_net_from_tf(sess, net, cost.network)
        # print('After move to TF: ')
        # net.print()

        # print_gradients(c, vars)

        # print(sess.run(cost.match_cost_n(1.)))
        # print(sess.run(cost.match_cost_w(1.)))
        # print(sess.run(cost.match_cost_e(1.)))
        # print(sess.run(cost.match_cost_f(1.)))
        # print(sess.run(cost.constraint_cost(1.)))

        update_net_from_tf(sess, net, cost.network)
        net.print()
        cost.compare_system(system, sess)
        # for i in range(len(net.connections)):
        # # for w in cost.network.w:
        #     print(net.connections)
        #     print(sess.run(w))

        print('optimize e')
        for i in range(10):
            optimize_net(sess, opt_op_e, clip_ops=clip_ops)
            cost_i = sess.run(c_e)
            print('cost: {}'.format(cost_i))

        update_net_from_tf(sess, net, cost.network)
        net.print()
        cost.compare_system(system, sess)

        print('optimize partial')
        for i in range(20):
            optimize_net(sess, opt_op_partial, clip_ops=clip_ops)
            cost_i = sess.run(c_partial)
            print('cost: {}'.format(cost_i))

        update_net_from_tf(sess, net, cost.network)
        net.print()
        cost.compare_system(system, sess)

        print('optimize full')
        cost_best = 1e10
        dc_dw = tf.gradients(c, cost.network.w)
        for i in range(250):
            # for d in dc_dw:
            #     print(sess.run(d))
            # for w in cost.network.w:
            #     print(sess.run(w))
            optimize_net(sess, opt_op, clip_ops=clip_ops)

            cost_i = sess.run(c)
            print('cost: {}'.format(cost_i))

            if cost_i < cost_best:
                cost_best = cost_i
                update_net_from_tf(sess, net, cost.network)

            if i % 10 == 0:
                print(sess.run(cost.match_cost_n(1.)))
                print(sess.run(cost.match_cost_w(1.)))
                print(sess.run(cost.match_cost_e(1.)))
                print(sess.run(cost.match_cost_f(1.)))
                print(sess.run(cost.match_cost_b(1.)))

            if np.isnan(cost_i):
                print('optimization failed')
                break

        # update_net_from_tf(sess, net, cost.network)
        net.print()

        cost.compare_system(system, sess)

        import pickle
        with open('net.pkl', 'wb') as netfile:
            pickle.dump(net, netfile)


        # print(sess.run(cost.match_cost_n(1.)))
        # print(sess.run(cost.match_cost_e(1.)))
        # print(sess.run(cost.match_cost_w(1.)))
        # print(sess.run(cost.match_cost_f(1.)))
        #
        # print('e values: ')
        # print(sess.run(cost._get_e_network(1)))
        # print(sess.run(cost._get_e_network(2)))
        #
        # # note gradient is None if there is no relation
        # print_gradients(c, vars)
        #
        # print('Variable values:')
        # for var in vars:
        #     print('{}: {}'.format(var.name, sess.run(var)))
