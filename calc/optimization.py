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


def test_stride_pattern(system, candidate=None):
    #TODO: add dead-end cost
    #TODO: remove w_rf or clip w_k somehow?
    if not candidate:
        candidate = StridePattern(system, 32)
        candidate.fill()

    net = initialize_network(system, candidate, image_layer=0, image_channels=3.)

    optimizer = tf.train.AdamOptimizer(learning_rate=.00001)
    print('Setting up cost structure')
    cost = Cost(system, net)

    print('Defining cost function')

    pc = cost.param_cost(1e-13)
    fc = cost.match_cost_f(1.)
    bc = cost.match_cost_b(1.)
    ec = cost.match_cost_e(1.)
    wc = cost.match_cost_w(1.)
    scc = cost.sparsity_constraint_cost(1.)
    kcc = cost.kernel_constraint_cost(1.)

    c = fc + bc + ec + wc
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
        # print('Printing')
        # update_net_from_tf(sess, net, cost.network)
        # net.print()

        training_curve = []
        c_value = sess.run(c)
        training_curve.append((0, c_value))
        _print_cost(c_value, None, None, sess.run(bc), sess.run(ec), None)

        iterations = 100
        for i in range(401):
            optimize_net(sess, opt_op, iterations=iterations, clip_ops=clip_ops)
            cost_i = sess.run(c)
            training_curve.append((iterations*i, cost_i))

            if np.isnan(cost_i):
                _print_cost(cost_i, sess.run(pc), sess.run(fc), sess.run(bc), sess.run(ec), sess.run(wc))
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
        print('.', end='', flush=True)

        opt_op.run()

        for clip_op in clip_ops:
            sess.run(clip_op)

    print()


def _print_cost(total_cost, param_cost, f_cost, b_cost, e_cost, rf_cost):
    print('total cost: {} param-cost: {} f cost: {} b cost {} e cost {} RF cost: {}'
          .format(total_cost, param_cost, f_cost, b_cost, e_cost, rf_cost))


