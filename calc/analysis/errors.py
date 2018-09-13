import os
import inspect
import csv
import numpy as np
import matplotlib.pyplot as plt
from calc.examples.example_systems import make_big_system


def get_layer_results(version='2'):
    filename = '{}/layer-results-{}.txt'.format(os.path.dirname(inspect.stack()[0][1]), version)

    with open(filename) as csvfile:
        r = csv.reader(csvfile)
        n = []
        e = []
        w_rf = []
        for row in r:
            vals = [_get_float(x) for x in row]
            # print(row)
            n.append([vals[0], vals[1]])
            e.append([vals[2], vals[3]])
            w_rf.append([vals[4], vals[5]])

    return np.array(n), np.array(e), np.array(w_rf)


def get_connection_results(version='2'):
    filename = '{}/connection-results-{}.txt'.format(os.path.dirname(inspect.stack()[0][1]), version)

    with open(filename) as csvfile:
        r = csv.reader(csvfile)
        b = []
        f = []
        for row in r:
            vals = [_get_float(x) for x in row]
            if vals[0] == 0:
                f.append([vals[1], vals[2]])
            else:
                b.append([vals[1], vals[2]])

    return np.array(b), np.array(f)

def _get_float(s):
    s = s.strip().split(';')[0]
    if s == 'None':
        return np.nan
    else:
        return float(s)


def get_results(version):
    n, e, w_rf = get_layer_results(version=version)
    b, f = get_connection_results(version=version)
    return {'n': n, 'e': e, 'w_rf': w_rf, 'b': b, 'f': f}


if __name__ == '__main__':
    one = get_results('fixed-f')
    two = get_results('fixed-g')

    # system = make_big_system()
    #
    # for i in range(len(system.populations)):
    #     pop = system.populations[i]
    #     print('{} {}'.format(pop.name, pop.e))

    # for i in range(len(system.populations)):
    #     if e[i,1]>5*e[i,0]:
    #         print('* ' + system.populations[i].name)
    #     else:
    #         print(system.populations[i].name)

    # plt.figure(figsize=(3.5,3))
    # plt.scatter(e[:,0], e[:,1], c=e[:,1]>10*e[:,0])
    # plt.tight_layout()
    # plt.savefig('e.eps')

    # import pickle
    # with open('../optimization-result-best-of-1000-c.pkl', 'rb') as file:
    #     data = pickle.load(file)
    # net = data['net']
    #
    # # the outlier is #94 ...
    # print(np.argmax(one['e'][:, 1]))
    # system = make_big_system()
    # print(system.populations[94].name)
    # pres = system.find_pre('46_4')
    # print(len(pres))
    # for pre in pres:
    #     conn_ind = system.find_projection_index(pre.name, '46_4')
    #     conn = net.connections[conn_ind]
    #     print('c: {} s: {}'.format(conn.c, conn.sigma))

    # in_degrees = []
    # for pop in system.populations:
    #     pre = system.find_pre(pop.name)
    #     in_degrees.append(len(pre))
    # plt.hist(in_degrees, 30)
    # plt.show()


    plt.figure(figsize=(3,2.5))
    plt.scatter(one['e'][:, 0], one['e'][:, 1], c='#A0A0A0', marker='^')
    plt.scatter(two['e'][:, 0], two['e'][:, 1], c='k', marker='+')
    plt.title('Extrinsic inputs')
    plt.tight_layout()
    plt.savefig('figures/e.eps')
    plt.show()

    plt.figure(figsize=(3,2.5))
    plt.scatter(one['n'][:, 0], one['n'][:, 1], c='#A0A0A0', marker='^')
    plt.scatter(two['n'][:, 0], two['n'][:, 1], c='k', marker='+')
    plt.title('Number of units')
    plt.tight_layout()
    plt.savefig('figures/n.eps')
    plt.show()

    plt.figure(figsize=(3,2.5))
    # plt.scatter(c['w_rf'][:,0], c['w_rf'][:,1])
    plt.scatter(one['w_rf'][:, 0], one['w_rf'][:, 1], c='#A0A0A0', marker='^')
    plt.scatter(two['w_rf'][:, 0], two['w_rf'][:, 1], c='k', marker='+')
    plt.title('RF width')
    plt.tight_layout()
    plt.savefig('figures/w_rf.eps')
    plt.show()

    # print('first connection {}->{}'.format(net.connections[6].pre.name, net.connections[6].post.name))
    # print(two['b'][:5,:])
    # foo = two['b'][:,1]
    # print(np.where(foo < 100))

    plt.figure(figsize=(3,2.5))
    # plt.scatter(c['b'][:,0], c['b'][:,1])
    plt.scatter(one['b'][:, 0], one['b'][:, 1], c='#A0A0A0', marker='^')
    plt.scatter(two['b'][:, 0], two['b'][:, 1], c='k', marker='+')
    plt.title('Inter-laminar in-degree')
    plt.tight_layout()
    plt.savefig('figures/b.eps')
    plt.show()

    plt.figure(figsize=(3,2.5))
    # plt.scatter(c['f'][:,0], c['f'][:,1])
    plt.scatter(one['f'][:, 0], one['f'][:, 1], c='#A0A0A0', marker='^')
    plt.scatter(two['f'][:, 0], two['f'][:, 1], c='k', marker='+')
    plt.title('FLNe')
    plt.tight_layout()
    plt.savefig('figures/f.eps')
    plt.show()

    print('f correlation: {}'.format(np.corrcoef(one['f'][:, 1], two['f'][:, 1])))
