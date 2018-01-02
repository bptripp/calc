import unittest
import tensorflow as tf
from calc.system import get_example_system, System
from calc.conversion import make_net_from_system, norm_squared_error, Cost, NetworkVariables
import numpy as np

def get_system_1():
    result = System()
    result.add_input(100, .02)
    result.add('V1', 100, 10, .1)
    result.connect_areas('INPUT', 'V1', 1.)
    return result


def get_system_2():
    result = System()
    result.add_input(100, .02)
    result.add('V1', 100, 10, .1)
    result.add('V2', 100, 10, .2)
    result.add('V4', 500, 10, .4)
    result.connect_areas('INPUT', 'V1', 1.)
    result.connect_areas('V1', 'V2', 1)
    result.connect_areas('V1', 'V4', .5)
    result.connect_areas('V2', 'V4', .5)
    return result


def get_cost_1():
    system = get_system_1()
    net = make_net_from_system(system)
    return system, net, Cost(system, net)


def get_cost_2():
    system = get_system_2()
    net = make_net_from_system(system)
    return system, net, Cost(system, net)


def calculate_n(net, ind):
    m_j = net.layers[ind].m
    width_j = net.layers[ind].width
    return m_j * width_j**2


def calculate_sigma_star(net, ind, w_ij, s_ij):
    post_ind = net.find_layer_index(net.connections[ind].post.name)

    c_ij = net.connections[ind].c
    sigma_ij = net.connections[ind].sigma
    m_i = net.layers[post_ind].m

    # print('c:{} sigma:{} w:{} m:{}'.format(c_ij, sigma_ij, w_ij, m_i))

    if s_ij < w_ij:
        beta_ij = (w_ij/s_ij)**2
    else:
        beta_ij = 1

    # print('stride: {} beta: {}'.format(s_ij, beta_ij))

    return 1. - (1. - sigma_ij)**(beta_ij * c_ij * m_i)


def calculate_n_ij(net, ind, s_ij, w_ij):
    # s_ij and w_ij don't come from the net, as the TF code treats it as a derived variable

    pre_ind = net.find_layer_index(net.connections[ind].pre.name)
    post_ind = net.find_layer_index(net.connections[ind].post.name)

    m_j = net.layers[pre_ind].m
    width_j = net.layers[pre_ind].width
    n_j = m_j * width_j**2
    c_ij = net.connections[ind].c
    sigma_ij = net.connections[ind].sigma
    m_i = net.layers[post_ind].m

    sigma_star_ij =  calculate_sigma_star(net, ind, w_ij, s_ij)
    # sigma_star_ij = 1. - (1. - sigma_ij)**(w_ij**2 * c_ij * m_i)

    if s_ij < w_ij:
        alpha_ij = 1
    else:
        alpha_ij = (w_ij/s_ij)**2

    result = n_j * c_ij * sigma_star_ij * alpha_ij
    return result


class TestNetworkVariables(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_init(self):
        system = get_system_1()
        net = make_net_from_system(system)
        nv = NetworkVariables(net, system, 0, system.populations[0].w)
        self.assertAlmostEqual(nv.image_pixel_width, system.populations[0].w)
        self.assertEqual(len(nv.width), 2)
        self.assertEqual(len(nv.w_rf), 2)
        self.assertEqual(len(nv.s), 1)
        self.assertEqual(nv.pres[0], 0)
        self.assertEqual(nv.posts[0], 1)
        self.assertEqual(nv.input_layers[1][0], 0)
        self.assertEqual(nv.input_connections[1][0], 0)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            self.assertAlmostEqual(sess.run(nv.w_rf[1]), .1)
            self.assertAlmostEqual(sess.run(nv.c[0]), net.connections[0].c)

            # these aren't taken directly from the network object but are derived
            self.assertNotAlmostEqual(sess.run(nv.w[0]), net.connections[0].w)
            self.assertNotAlmostEqual(sess.run(nv.s[0]), net.connections[0].s)


class TestCost(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_get_n_network(self):
        system, net, cost = get_cost_1()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            result = sess.run(cost._get_n_network(1))
            answer = net.layers[1].m * net.layers[1].width**2
            self.assertAlmostEqual(result, answer, 3)

    def test_norm_squared_error(self):
        target = tf.Variable(2.)
        actual = tf.Variable(1.)
        e = norm_squared_error(target, actual)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            self.assertAlmostEqual(sess.run(e), 0.4804530139)

    def test_match_cost_n(self):
        system, net, cost = get_cost_1()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            result = sess.run(cost.match_cost_n(1.))

            answer = 0
            for i in range(len(system.populations)):
                system_n = system.populations[i].n
                net_n = net.layers[i].width**2 * net.layers[i].m
                err = np.log(net_n / system_n)**2
                # err = ((net_n - system_n) / system_n)**2
                # print('PY sys: {} net: {} err: {}'.format(system_n, net_n, err))
                # e = norm_squared_error(cost.system.n[i], cost.get_n_network(i))
                # print('TF sys: {} net: {} err: {}'.format(sess.run(cost.system.n[i]), sess.run(cost.get_n_network(i)), sess.run(e)))

                answer = answer + err

            answer = answer / len(system.populations)

            self.assertAlmostEqual(result, answer)

    def test_match_cost_e(self):
        system, net, cost = get_cost_1()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            system_e = system.populations[1].e
            # just one input here, so no sum ...
            w = sess.run(cost.network.w[0])  # the TF node isn't taken from the net but calculated as a function of RFs
            net_e = w**2 * net.connections[0].c * net.layers[0].m * net.connections[0].sigma
            answer = np.log(net_e / system_e)**2
            # answer = ((net_e - system_e) / system_e)**2
            result = sess.run(cost.match_cost_e(1.))
            self.assertAlmostEqual(result, answer, 3)

    def test_match_cost_w(self):
        system, net, cost = get_cost_2()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            answer = 0.
            for i in range(len(system.populations)):
                sys_w_rf = system.populations[i].w
                net_w_rf = sess.run(cost.network.w_rf[i])
                err = ((sys_w_rf - net_w_rf)/sys_w_rf)**2
                answer = answer + err

            result = sess.run(cost.match_cost_w(1.))
            self.assertAlmostEqual(result, answer)

    def test_sigma_star(self):
        system, net, cost = get_cost_2()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            for conn_ind in range(len(net.connections)):
                tf_sigma_star = cost.sigma_star(conn_ind)

                tf_w = sess.run(cost.network.w[conn_ind])
                tf_s = sess.run(cost.network.s[conn_ind])
                net_sigma_star = calculate_sigma_star(net, conn_ind, tf_w, tf_s)
                # print('sigma_star net: {} tf: {}'.format(net_sigma_star, sess.run(tf_sigma_star)))

                self.assertAlmostEqual(net_sigma_star, sess.run(tf_sigma_star), 3)

    def test_match_cost_f(self):
        # system, net, cost = get_cost_1()
        system, net, cost = get_cost_2()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            answer = 0.
            for i in range(len(system.projections)):
                sys_f = system.projections[i].f

                s_ij = sess.run(cost.network.s[i])
                w_ij = sess.run(cost.network.w[i])
                net_n = calculate_n_ij(net, i, s_ij, w_ij)

                net_n_total = 0.
                for j in range(len(net.connections)):
                    if net.connections[j].post.name == system.projections[i].termination.name:
                        s_ij = sess.run(cost.network.s[j])
                        w_ij = sess.run(cost.network.w[j])
                        net_n_total = net_n_total + calculate_n_ij(net, j, s_ij, w_ij)

                net_f = net_n / net_n_total

                err = np.log(net_f / sys_f) ** 2
                # err = ((sys_f - net_f) / sys_f)**2
                # print('net_n: {} net_f: {} sys_f: {} err: {}'.format(net_n, net_f, sys_f, err))
                answer = answer + err

            answer = answer / len(system.projections)

            result = sess.run(cost.match_cost_f(1.))
            self.assertAlmostEqual(result, answer, 3)

    def test_kappas(self):
        system, net, cost = get_cost_1()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            answer = 2.*sess.run(cost.match_cost_n(1.)) \
                     + 4.*sess.run(cost.match_cost_e(1.)) \
                     + 6.*sess.run(cost.match_cost_w(1.)) \
                     + 8.*sess.run(cost.match_cost_f(1.)) \
                     + 10.*sess.run(cost.param_cost(1.))
            result = sess.run(cost.match_cost_n(2.)) \
                     + sess.run(cost.match_cost_e(4.)) \
                     + sess.run(cost.match_cost_w(6.)) \
                     + sess.run(cost.match_cost_f(8.)) \
                     + sess.run(cost.param_cost(10.))
            self.assertAlmostEqual(result, answer, 2)

    def test_param_cost_1(self):
        system1, net1, cost1 = get_cost_1()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            w = sess.run(cost1.network.w[0]) # the TF node isn't taken from the net but calculated as a function of RFs
            answer = w**2 * net1.layers[0].m * net1.layers[1].m
            result = sess.run(cost1.param_cost(1.))
            self.assertAlmostEqual(result, answer, 3)


def test_param_cost_2(self):
    system2, net2, cost2 = get_cost_2()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        w0 = sess.run(cost2.network.w[0])
        answer0 = w0 ** 2 * net2.layers[0].m * net2.layers[1].m
        w1 = sess.run(cost2.network.w[1])
        answer1 = w1 ** 2 * net2.layers[1].m * net2.layers[2].m
        w2 = sess.run(cost2.network.w[2])
        answer2 = w2 ** 2 * net2.layers[1].m * net2.layers[3].m
        w3 = sess.run(cost2.network.w[3])
        answer3 = w3 ** 2 * net2.layers[2].m * net2.layers[3].m
        answer = answer0 + answer1 + answer2 + answer3
        result = sess.run(cost2.param_cost(1.))
        self.assertAlmostEqual(result, answer, 2)


if __name__ == '__main__':
    unittest.main()