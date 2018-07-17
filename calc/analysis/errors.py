import os
import inspect
import csv
import numpy as np
import matplotlib.pyplot as plt
from calc.examples.example_systems import make_big_system

def get_layer_results():
    filename = '{}/layer-results-2.txt'.format(os.path.dirname(inspect.stack()[0][1]))

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


def get_connection_results():
    filename = '{}/connection-results-2.txt'.format(os.path.dirname(inspect.stack()[0][1]))

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


if __name__ == '__main__':
    n, e, w_rf = get_layer_results()
    b, f = get_connection_results()

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

    plt.figure(figsize=(3.5,3))
    plt.scatter(e[:,0], e[:,1])
    plt.tight_layout()
    plt.savefig('e.eps')

    # plt.figure(figsize=(3.5,3))
    # plt.scatter(n[:,0], n[:,1])
    # plt.tight_layout()
    # plt.savefig('n.eps')
    #
    # plt.figure(figsize=(3.5,3))
    # plt.scatter(w_rf[:,0], w_rf[:,1])
    # plt.tight_layout()
    # plt.savefig('w_rf.eps')
    #
    # plt.figure(figsize=(3.5,3))
    # plt.scatter(b[:,0], b[:,1])
    # plt.tight_layout()
    # plt.savefig('b.eps')
    #
    # plt.figure(figsize=(3.5,3))
    # plt.scatter(f[:,0], f[:,1])
    # plt.tight_layout()
    # plt.savefig('f.eps')

    # plt.show()