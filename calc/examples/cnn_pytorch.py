"""
Builds a PyTorch model from a CALC Network.
"""

import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Backbone(nn.Module):
    """
    A network that is modelled on part of primate visual cortex.
    """

    def __init__(self, network, layer_names, output, c_scale=1.0, sigma_scale=1.0, random_seed=1):
        super(Backbone, self).__init__()

        np.random.seed(seed=random_seed)
        subsample_indices = preprocess(network, c_scale, sigma_scale, output)

        self.layer_names = layer_names
        self._output_index = self.layer_names.index(output)

        self.bns = nn.ModuleList()
        self.inbound_connection_inds = []
        for layer_name in self.layer_names:
            layer = network.find_layer(layer_name)
            self.bns.append(nn.BatchNorm2d(layer.m))
            in_connections = network.find_inbounds(layer_name)
            in_indices = [network.find_connection_index(i.pre.name, i.post.name) for i in in_connections]
            self.inbound_connection_inds.append(in_indices)

        self.sigmas = []
        self.masks = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.pre_layer_inds = []
        self.subsample_indices = nn.ParameterList()
        for i, connection in enumerate(network.connections):
            self.sigmas.append(connection.sigma)
            padding = _get_padding(connection.w)
            input_channels = len(subsample_indices[i])
            conv = nn.Conv2d(input_channels, connection.post.m, connection.w, connection.s, padding=padding)
            self.convs.append(conv)
            self.pre_layer_inds.append(self.layer_names.index(connection.pre.name))
            self.subsample_indices.append(nn.Parameter(torch.LongTensor(subsample_indices[i]), requires_grad=False))

    def keep_sparse(self):
        for mask, conv in zip(self.masks, self.convs):
            conv.weight.requires_grad = False
            conv.weight *= mask.float()
            conv.weight.requires_grad = True

    def forward(self, input):
        activities = []
        for i, layer_name in enumerate(self.layer_names):
            if layer_name == 'INPUT':
                activities.append(input)
            else:
                components = []
                for connection_ind in self.inbound_connection_inds[i]:
                    pre_layer_ind = self.pre_layer_inds[connection_ind]
                    sub_ind = self.subsample_indices[connection_ind]
                    conv = self.convs[connection_ind]
                    input = activities[pre_layer_ind].index_select(1, sub_ind)
                    components.append(conv(input))

                x = torch.stack(components).mean(dim=0)
                bn = self.bns[i]
                x = F.relu(bn(x))
                activities.append(x)

        return activities[self._output_index]

    def print_architecture(self):
        for i, layer_name in enumerate(self.layer_names):
            print('{} {} channels'.format(layer_name, len(self.bns[i].weight)))

        def find_post_index(connection_index):
            for i, ic in enumerate(self.inbound_connection_inds):
                if connection_index in ic:
                    return i

        for i in range(len(self.convs)):
            pre_index = self.pre_layer_inds[i]
            post_index = find_post_index(i)

            c = self.convs[i].weight.shape[1] / len(self.bns[pre_index].weight)
            n_zeros = np.sum((self.convs[i].weight == 0).numpy())
            n_elements = np.prod(self.convs[i].weight.shape)

            print('{}->{} c={:.6} sigma={:.6} stride={} w={}'.format(
                self.layer_names[pre_index],
                self.layer_names[post_index],
                c,
                (n_elements - n_zeros) / n_elements,
                self.convs[i].stride,
                self.convs[i].weight.shape[2]
            ))


class Cifar10Classifier(nn.Module):
    def __init__(self, backbone):
        super(Cifar10Classifier, self).__init__()

        self.backbone = backbone
        backbone_channels = backbone.forward(torch.Tensor(np.random.randn(2, 3, 32, 32))).shape[1]

        self.conv1 = nn.Conv2d(backbone_channels, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.avg = nn.AvgPool2d(2)
        self.linear1 = nn.Linear(64, 512)
        self.drop1 = nn.Dropout(.5)
        self.linear2 = nn.Linear(512, 512)
        self.drop2 = nn.Dropout(.25)
        self.classifier = nn.Linear(512, 10) #TODO: from logits

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.drop1(x)
        x = F.relu(self.linear2(x))
        x = self.drop2(x)
        x = self.classifier(x)
        return x

    def snip(self, batches=10):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

        criterion = nn.CrossEntropyLoss()

        weights = []
        grads = []
        for conv in self.backbone.convs:
            weights.append(conv.weight)
            grads.append(torch.Tensor(np.zeros(conv.weight.shape)))

        for batch_idx, (inputs, targets) in enumerate(testloader):
            if batch_idx == batches:
                break
            print('SNIP batch {} of {}'.format(batch_idx, batches))

            outputs = self(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for i, conv in enumerate(self.backbone.convs):
                grads[i] += conv.weight.grad

        self.backbone.masks = nn.ParameterList()
        for i in range(len(self.backbone.convs)):
            absolute_sensitivities = torch.abs(weights[i] * grads[i])
            n_to_keep = int(absolute_sensitivities.numel() * self.backbone.sigmas[i])

            threshold = np.sort(absolute_sensitivities.detach().numpy().flatten())[-n_to_keep]
            mask = nn.Parameter(absolute_sensitivities >= threshold, requires_grad=False)
            self.backbone.masks.append(mask)


class ImageNetClassifier(nn.Module):
    def __init__(self, backbone):
        super(ImageNetClassifier, self).__init__()

        self.backbone = backbone
        backbone_channels = backbone.forward(torch.Tensor(np.random.randn(2, 3, 32, 32))).shape[1]
        print('backbone channels: {}'.format(backbone_channels))

        self.conv1 = nn.Conv2d(backbone_channels, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.avg = nn.AvgPool2d(2)
        self.linear1 = nn.Linear(64*7*7, 2048) #change based on output size
        self.drop1 = nn.Dropout(.5)
        self.classifier = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.drop1(x)
        x = self.classifier(x)
        return x


def _get_padding(w):
    padding = (w - 1) / 2
    c = int(np.ceil(padding))
    return c


def subsample_maps(net):
    """
    Randomly samples presynaptic feature maps that contribute to each connection.
    We try to make sets of maps of a given layer that contribute to different
    interarea connections mostly disjoint, although this doesn't have to be done
    strictly, and in any case it's not always possible as the total across connections
    may be greater than the number of maps. We do the same for interlaminar connections.

    :param net: a Network
    :return: indices of presynaptic feature maps to use in each connection
    """

    # start with uniform probability that each map is selected for inclusion in a connection;
    # reduce probability of selecting a given map again whenever it is selected
    probability_reduction = .01
    interarea_probabilities = []
    interlaminar_probabilities = []
    for layer in net.layers:
        interarea_probabilities.append(np.ones(layer.m))
        interlaminar_probabilities.append(np.ones(layer.m))

    subsample_indices = []
    for connection in net.connections:
        n_maps = int(round(connection.pre.m * connection.c))
        pre_index = net.find_layer_index(connection.pre.name)

        pre_cortical_area = connection.pre.name.split('_')[0]
        post_cortical_area = connection.post.name.split('_')[0]
        if pre_cortical_area == post_cortical_area: # interlaminar
            p = interlaminar_probabilities[pre_index]
        else:
            p = interarea_probabilities[pre_index]

        indices = np.random.choice(range(int(connection.pre.m)), n_maps, replace=False, p=p/np.sum(p))
        for i in indices:
            p[i] = p[i]*probability_reduction
        subsample_indices.append(indices)

    return subsample_indices


def prune_maps(net, subsample_indices, output_name):
    """
    In case not all the feature maps are used in feedforward connections, the unused ones
    can optionally be omitted from a model, if the model is not to include feedback
    or lateral connections.

    :param net:
    :param subsample_indices: result of subsample_maps; indices of maps that provide input to each connection
    :param output_name: name of layer used to feed output (not pruned)
    :return:
    """

    # find which feature maps are used in output
    indices_by_layer = []
    for layer in net.layers:
        connections = net.find_outbounds(layer.name)

        all_indices_for_layer = []
        for connection in connections:
            conn_ind = net.find_connection_index(connection.pre.name, connection.post.name)
            all_indices_for_layer.extend(subsample_indices[conn_ind])

        all_indices_for_layer = list(set(all_indices_for_layer))
        all_indices_for_layer.sort()
        indices_by_layer.append(all_indices_for_layer)

    new_subsample_indices = copy.deepcopy(subsample_indices)

    # discard unused maps and condense indices
    for i in range(len(net.layers)):
        layer = net.layers[i]

        if not layer.name == output_name and not layer.name == 'INPUT':
            connections = net.find_outbounds(layer.name)

            for connection in connections:
                ind = net.find_connection_index(connection.pre.name, connection.post.name)
                for j in range(len(new_subsample_indices[ind])):
                    new_subsample_indices[ind][j] = indices_by_layer[i].index(subsample_indices[ind][j])

            old_m = layer.m
            layer.m = len(indices_by_layer[i])

    return new_subsample_indices


def prune_connections(net, subsample_indices):
    """
    Remove connections with no subsample indices.

    :param net:
    :param subsample_indices:
    :return:
    """
    new_connections = []
    new_subsample_indices = []
    for i in range(len(subsample_indices)):
        if len(subsample_indices[i]) > 0:
            new_connections.append(net.connections[i])
            new_subsample_indices.append(subsample_indices[i])

    net.connections = new_connections
    return new_subsample_indices


def prune_layers(net, subsample_indices):
    """
    Remove layers with no maps and their associated connections. It is possible
    for a layers to lose all its maps in prune_maps().

    :param net: a Network
    :return: updated Network with empty layers and associated connections removed
    """
    new_layers = []

    discarded_names = []
    for layer in net.layers:
        if layer.m > 0:
            new_layers.append(layer)
        else:
            discarded_names.append(layer.name)

    new_connections = []
    new_subsample_indices = []
    for i in range(len(net.connections)):
        connection = net.connections[i]
        if connection.pre.name not in discarded_names and connection.post.name not in discarded_names:
            new_connections.append(connection)
            new_subsample_indices.append(subsample_indices[i])

    net.layers = new_layers
    net.connections = new_connections

    return new_subsample_indices


def preprocess(net, c_scale, sigma_scale, output):
    for layer in net.layers:
        layer.m = int(round(layer.m))

    def set_c(connection, new_c):
        connection.sigma = connection.sigma * connection.c / new_c
        connection.c = new_c

    for connection in net.connections:
        if 'INPUT' in connection.pre.name or 'LGN' in connection.pre.name:
            set_c(connection, 1)

    alpha_total = 0
    beta_total = 0
    for connection in net.connections:
        if connection.pre.name == 'V1_4Calpha':
            alpha_total += connection.c * connection.sigma
        if connection.pre.name == 'V1_4Cbeta':
            beta_total += connection.c * connection.sigma

    for connection in net.connections:
        if connection.pre.name == 'V1_4Calpha':
            fraction = connection.c * connection.sigma / alpha_total
            set_c(connection, fraction)
        if connection.pre.name == 'V1_4Cbeta':
            fraction = connection.c * connection.sigma / beta_total
            set_c(connection, fraction)

    net.scale_c(c_scale)
    net.scale_sigma(sigma_scale)

    subsample_indices = subsample_maps(net)
    subsample_indices = prune_maps(net, subsample_indices, output)
    subsample_indices = prune_connections(net, subsample_indices)
    subsample_indices = prune_layers(net, subsample_indices)

    # keep track of subsample indices by name, because connection indices may change
    subsample_map = {}
    for i in range(len(net.connections)):
        subsample_map[net.connections[i].get_name()] = subsample_indices[i]

    removed_indices = net.prune_dead_ends([output])
    removed_indices = np.sort(removed_indices)
    for i in range(len(removed_indices)):
        next_largest_removed_index = removed_indices[-1-i]
        del subsample_indices[next_largest_removed_index]

    subsample_indices = []
    for i in range(len(net.connections)):
        subsample_indices.append(subsample_map[net.connections[i].get_name()])

    for layer in net.layers:
        if layer.m < 1:
            print('WARNING: {} has {} maps'.format(layer.name, layer.m))

    for connection in net.connections:
        connection.s = int(round(connection.s))
        connection.w = int(round(connection.w))
        if connection.w % 2 == 0: # https://github.com/pytorch/pytorch/issues/3867
            connection.w += 1

        if connection.w < 1:
            print('setting w to 1 for {}->{}'.format(connection.pre.name, connection.post.name))
            connection.w = 1

    return subsample_indices


def load_model(opt_file='network_structure.pkl', checkpoint_file='trained_params.tar'):
    with open(opt_file, 'rb') as file:
        data = pickle.load(file)
    net = data['net']
    layer_names = data['layer_names']

    backbone = Backbone(net, layer_names, 'PITv_2/3', c_scale=.5, sigma_scale=.5)
    net = Cifar10Classifier(backbone)
    net.snip(batches=1)

    net = ImageNetClassifier(backbone)
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    print('Trained on ImageNet for {} epochs, top-1 accuracy {}'.format(
        checkpoint['epoch'], checkpoint['best_acc1']))
    net.load_state_dict(checkpoint['state_dict'])
    return net


if __name__ == '__main__':
    net = load_model()
    net.backbone.print_architecture()
