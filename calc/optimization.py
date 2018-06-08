import pickle
import numpy as np
import tensorflow as tf
from calc.stride import StridePattern, initialize_network
from calc.conversion import get_clip_ops, update_net_from_tf, make_net_from_system, Cost

"""
Optimization strategies.
"""


def test_stride_patterns(system, n=5):
    import matplotlib.pyplot as plt

    nets = [None] * n
    training_curves = [None] * n
    for i in range(n):
        tf.reset_default_graph()
        nets[i], training_curves[i] = test_stride_pattern(system)
        tc = np.array(training_curves[i])
        print(tc.shape)
        plt.semilogy(tc[:,0], tc[:,1])

    data = {
        'training_curves': training_curves,
        'nets': nets
    }
    with open('calc-training.pickle', 'wb') as f:
        pickle.dump(data, f)

    plt.show()


def test_stride_pattern(system):
    candidate = StridePattern(system, 32)
    candidate.fill()
    net = initialize_network(system, candidate, image_layer=0, image_channels=3.)

    optimizer = tf.train.AdamOptimizer()
    print('Setting up cost structure')
    cost = Cost(system, net)

    print('Defining cost function')
    c = cost.match_cost_f(1.) \
        + cost.match_cost_b(1.) \
        + cost.match_cost_e(1.) \
        + cost.match_cost_w(1.) \
        + cost.param_cost(1e-11) \
        + cost.sparsity_constraint_cost(1.)

    pc = cost.param_cost(1.)
    fc = cost.match_cost_f(1.)
    bc = cost.match_cost_b(1.)
    ec = cost.match_cost_e(1.)
    wc = cost.match_cost_w(1.)

    vars = []
    vars.extend(cost.network.c)
    vars.extend(cost.network.sigma)
    for w_rf in cost.network.w_rf:
        if isinstance(w_rf, tf.Variable):
            vars.append(w_rf)

    clip_ops = []
    clip_ops.extend(get_clip_ops(cost.network.c))
    clip_ops.extend(get_clip_ops(cost.network.sigma))

    print('Setting up optimizer')
    opt_op = optimizer.minimize(c, var_list=vars)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        print('Initializing')
        sess.run(init)
        print('Printing')
        update_net_from_tf(sess, net, cost.network)
        net.print()

        training_curve = []
        training_curve.append((0, sess.run(c), sess.run(pc), sess.run(wc)))

        _print_cost(sess.run(c), sess.run(pc), sess.run(fc), sess.run(bc), sess.run(ec), sess.run(wc))

        iterations = 100
        for i in range(201):
            optimize_net(sess, opt_op, iterations=iterations, clip_ops=clip_ops)
            cost_i = sess.run(c)
            cost_p_i = sess.run(pc)
            cost_w_i = sess.run(wc)
            _print_cost(cost_i, cost_p_i, sess.run(fc), sess.run(bc), sess.run(ec), cost_w_i)
            training_curve.append((iterations*i, cost_i, cost_p_i, cost_w_i))

            if np.isnan(cost_i):
                break

        update_net_from_tf(sess, net, cost.network)
        net.print()
        cost.compare_system(system, sess)

    return net, training_curve


def optimize_net(sess, opt_op, iterations=100, clip_ops=[]):
    # gradients = tf.gradients(c, vars)
    for i in range(iterations):
        # for i in range(len(vars)):
        #     if gradients[i] is not None:
        #         print('grad wrt {}: {}'.format(vars[i].name, sess.run(gradients[i])))
        opt_op.run()

        for clip_op in clip_ops:
            sess.run(clip_op)


def _print_cost(total_cost, param_cost, f_cost, b_cost, e_cost, rf_cost):
    print('total cost: {} param-cost: {} f cost: {} b cost {} e cost {} RF cost: {}'
          .format(total_cost, param_cost, f_cost, b_cost, e_cost, rf_cost))


def make_optimal_network(system):
    net = make_net_from_system(system)
    cost = Cost(system, net)
    # TODO: don't need n in optimization
    c = cost.match_cost_n(1.) \
        + cost.match_cost_w(1.) \
        + cost.match_cost_e(1.) \
        + cost.match_cost_f(1.) \
        + cost.match_cost_b(1.) \
        + cost.constraint_cost(1.)

    optimizer = tf.train.AdamOptimizer()
    vars = cost.network.collect_variables()
    opt_op = optimizer.minimize(c, var_list=vars)

    # opt_fast = tf.train.AdamOptimizer()
    c_e = cost.match_cost_e(1.)
    opt_op_e = optimizer.minimize(c_e, var_list=cost.network.sigma)

    c_partial = cost.match_cost_f(1.) + cost.match_cost_e(1.)
    vars_partial = []
    vars_partial.extend(cost.network.c)
    vars_partial.extend(cost.network.sigma)
    opt_op_partial = optimizer.minimize(c_partial, var_list=vars_partial)

    clip_ops = []
    clip_ops.extend(get_clip_ops(cost.network.c))
    clip_ops.extend(get_clip_ops(cost.network.sigma))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        update_net_from_tf(sess, net, cost.network)

        # net.print()
        # cost.compare_system(system, sess)

        print('optimize e')
        for i in range(10):
            optimize_net(sess, opt_op_e, clip_ops=clip_ops)
            cost_i = sess.run(c_e)
            print('cost: {}'.format(cost_i))

        update_net_from_tf(sess, net, cost.network)
        net.print()
        cost.compare_system(system, sess)

        print('optimize f and e')
        for i in range(20):
            optimize_net(sess, opt_op_partial, clip_ops=clip_ops)
            cost_i = sess.run(c_partial)
            print('cost: {}'.format(cost_i))

        update_net_from_tf(sess, net, cost.network)
        net.print()
        cost.compare_system(system, sess)

        print('optimize full')
        cost_best = 1e10
        for i in range(20): #TODO: more iterations here
            optimize_net(sess, opt_op, clip_ops=clip_ops)

            cost_i = sess.run(c)
            print('cost: {}'.format(cost_i))

            if cost_i < cost_best:
                cost_best = cost_i
                update_net_from_tf(sess, net, cost.network)

            if i % 20 == 0:
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
        return net


