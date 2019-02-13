import pickle
import numpy as np
import tensorflow as tf
from calc.stride import StridePattern, initialize_network
from calc.conversion import get_clip_ops, update_net_from_tf, Cost


def compare_stride_patterns(system, n=5):
    """
    Creates and optimizes networks with several random stride patterns, to
    explore sensitivity of optimization to stride pattern.

    :param system: a System (physiological model) for which to create a convnet
    :param n: number of stride patterns / networks to generate
    """
    import matplotlib.pyplot as plt

    nets = [None] * n
    training_curves = [None] * n
    for i in range(n):
        tf.reset_default_graph()
        nets[i], training_curves[i] = optimize_network_architecture(system)
        tc = np.array(training_curves[i])
        print(tc.shape)
        plt.semilogy(tc[:,0], tc[:,1])

    data = {
        'training_curves': training_curves,
        'nets': nets
    }
    with open('nets-and-training-curves.pkl', 'wb') as f:
        pickle.dump(data, f)

    plt.show()


def optimize_network_architecture(system, stride_pattern=None, compare=True):
    """
    :param system: a System (physiological model) to fit network architecture to
    :param stride_pattern: strides for each connection (generated at random if not given
    :param compare: print comparison of optimized values with target values
    :return: optimized network, training curve
    """

    if not stride_pattern:
        stride_pattern = StridePattern(system, 32)
        stride_pattern.fill()

    net = initialize_network(system, stride_pattern, image_layer=0, image_channels=3.)

    optimizer = tf.train.AdamOptimizer(learning_rate=.00001)
    print('Setting up cost structure')
    cost = Cost(system, net)

    print('Defining cost function')

    # pc = cost.param_cost(1e-13)
    fc = cost.match_cost_f(1.)
    bc = cost.match_cost_b(1.)
    ec = cost.match_cost_e(1.)
    wc = cost.match_cost_w(1.)
    dec = cost.dead_end_cost(1.)
    kc = cost.w_k_constraint_cost(1.)

    c = fc + bc + ec + wc + dec + kc

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
        c_value = sess.run(c)
        training_curve.append((0, c_value, sess.run(dec), sess.run(kc)))
        _print_cost(c_value, sess.run(fc), sess.run(bc), sess.run(ec), sess.run(wc), sess.run(dec))

        iterations = 100
        for i in range(1001):
            _run_optimization_steps(sess, opt_op, iterations=iterations, clip_ops=clip_ops)
            cost_i = sess.run(c)
            training_curve.append((iterations*i, cost_i, sess.run(dec), sess.run(kc)))
            print(cost_i)

            if np.isnan(cost_i):
                _print_cost(cost_i, sess.run(fc), sess.run(bc), sess.run(ec), sess.run(wc), sess.run(dec))
                break

            if i > 0 and i % 50 == 0:
                print('saving checkpoint')
                update_net_from_tf(sess, net, cost.network)
                with open('optimization-checkpoint.pkl', 'wb') as file:
                    pickle.dump({'net': net, 'training_curve': training_curve}, file)

        update_net_from_tf(sess, net, cost.network)
        net.print()

        if compare:
            cost.compare_system(system, sess)

    return net, training_curve


def _run_optimization_steps(sess, opt_op, iterations=100, clip_ops=[]):
    for i in range(iterations):
        print('.', end='', flush=True)
        opt_op.run()
        for clip_op in clip_ops:
            sess.run(clip_op)
    print()


def _print_cost(total_cost, f_cost, b_cost, e_cost, rf_cost, de_cost):
    print('total cost: {} f cost: {} b cost {} e cost {} RF cost: {} dead-end cost: {}'
          .format(total_cost, f_cost, b_cost, e_cost, rf_cost, de_cost))

