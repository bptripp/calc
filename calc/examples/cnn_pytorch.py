"""
For building a PyTorch model from a CALC Network.
"""

# MAYBE: create, save & reload, compare Keras model
# DONE: subsample view
# DONE: SNIP
# DONE: sparsity constraint
# TODO later: scaled Glorot
# DONE: use nn.ModuleList instead of setattr

import pickle
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from util import subsample_maps, prune_maps, prune_layers, prune_connections


class Backbone(nn.Module):
    def __init__(self, network, output, c_scale=1.0, sigma_scale=1.0, random_seed=1):
        super(Backbone, self).__init__()

        np.random.seed(seed=random_seed)
        subsample_indices = preprocess(network, c_scale, sigma_scale, output)

        graph = network.make_graph()
        self.layer_names = [name for name in nx.topological_sort(graph)]
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
        # self.subsample_indices = []
        self.subsample_indices = nn.ParameterList()
        for i, connection in enumerate(network.connections):
            self.sigmas.append(connection.sigma)
            padding = _get_padding(connection.w)
            input_channels = len(subsample_indices[i])
            conv = nn.Conv2d(input_channels, connection.post.m, connection.w, connection.s, padding=padding)
            self.convs.append(conv)
            self.pre_layer_inds.append(self.layer_names.index(connection.pre.name))
            # self.subsample_indices.append(torch.LongTensor(subsample_indices[i]))
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

                # print('******')
                # for component in components:
                #     print(component.shape)

                x = torch.stack(components).mean(dim=0)
                bn = self.bns[i]
                # bn = getattr(self, _bn_name(layer_name))
                x = F.relu(bn(x))
                activities.append(x)

        return activities[self._output_index]

    def print(self):
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

            # print(self.convs[i].weight.shape)
            # w = self.convs[i].weight.shape[2]
            # print(w)

            # print('zeros: {}/{}'.format(n_zeros, n_elements))
            print('{}->{} c={:.6} sigma={:.6} stride={} w={}'.format(
                self.layer_names[pre_index],
                self.layer_names[post_index],
                c,
                (n_elements - n_zeros) / n_elements,
                self.convs[i].stride,
                self.convs[i].weight.shape[2]
            ))


class Classifier(nn.Module):
    def __init__(self, backbone):
        super(Classifier, self).__init__()

        self.backbone = backbone
        backbone_channels = backbone.forward(torch.Tensor(np.random.randn(2, 3, 32, 32))).shape[1]
        print(backbone.forward(torch.Tensor(np.random.randn(2, 3, 32, 32))).shape)
        print('backbone channels: {}'.format(backbone_channels))

        self.conv1 = nn.Conv2d(backbone_channels, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.avg = nn.AvgPool2d(2)
        self.linear1 = nn.Linear(64, 512)
        self.drop1 = nn.Dropout(.5)
        self.linear2 = nn.Linear(512, 512)
        self.drop2 = nn.Dropout(.25)
        self.classifier = nn.Linear(512, 10) #TODO: from logits
        # self.snip()

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

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
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
        print(backbone.forward(torch.Tensor(np.random.randn(2, 3, 32, 32))).shape)
        print('backbone channels: {}'.format(backbone_channels))

        self.conv1 = nn.Conv2d(backbone_channels, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.avg = nn.AvgPool2d(2)
        self.linear1 = nn.Linear(64*7*7, 2048) #change based on output size
        self.drop1 = nn.Dropout(.5)
        # self.linear2 = nn.Linear(2048, 2048)
        # self.drop2 = nn.Dropout(.25)
        self.classifier = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.avg(x)
        x = x.view(x.size(0), -1) #[1 x 3136], m2: [64 x 2048]  # [1 x 12544], m2: [3136 x 2048]
        x = F.relu(self.linear1(x))
        x = self.drop1(x)
        # x = F.relu(self.linear2(x))
        # x = self.drop2(x)
        x = self.classifier(x)
        return x


def _get_padding(w):
    padding = (w - 1) / 2
    c = int(np.ceil(padding))
    return c


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

    # for i in range(len(subsample_indices)):
    #     print('{}->{}: {}'.format(net.connections[i].pre.name, net.connections[i].post.name, len(subsample_indices[i])))

    removed_indices = net.prune_dead_ends([output])
    removed_indices = np.sort(removed_indices)
    for i in range(len(removed_indices)):
        next_largest_removed_index = removed_indices[-1-i]
        del subsample_indices[next_largest_removed_index]

    subsample_indices = []
    for i in range(len(net.connections)):
        subsample_indices.append(subsample_map[net.connections[i].get_name()])

    # for i in range(len(subsample_indices)):
    #     print('{}->{}: {}'.format(net.connections[i].pre.name, net.connections[i].post.name, len(subsample_indices[i])))

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

    # input_layer = net.find_layer('INPUT')
    # # input_channels = int(input_layer.m)
    # input_channels = input_layer.m

    return subsample_indices


def get_activities(checkpoint_file, stimulus_directory, opt_file='optimization-result-PITv.pkl', result_file=None):
    with open(opt_file, 'rb') as file:
        data = pickle.load(file)
    net = data['net']
    backbone = Backbone(net, 'PITv_2/3', c_scale=.5, sigma_scale=.5)
    net = Classifier(backbone)
    net.snip(batches=1)

    # CIFAR-10
    # checkpoint = torch.load(checkpoint_file, map_location='cpu')
    # net.load_state_dict(checkpoint['net'])

    # ImageNet
    net = ImageNetClassifier(backbone)
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    # print(checkpoint['net'])
    net.load_state_dict(checkpoint['state_dict'])
    # net.load_state_dict(checkpoint['net'])

    stim_dataset = torchvision.datasets.ImageFolder(
        root=stimulus_directory,
        transform=torchvision.transforms.ToTensor()
    )
    # print(stim_dataset.imgs) # alphabetical by folder file name

    stim_loader = torch.utils.data.DataLoader(
        stim_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False
    )

    activities = []
    categories = []
    for batch_idx, (inputs, targets) in enumerate(stim_loader):
        print('Processing stimulus {} of {}'.format(batch_idx, len(stim_loader)))
        activities.append(backbone(inputs).detach().numpy())
        categories.append(targets.numpy()[0]) # note batch size 1

    if result_file is not None:
        with open(result_file, 'wb') as f:
            pickle.dump((activities, categories), f)

    return activities, categories

if __name__ == '__main__':
    # with open('optimization-result-PITv.pkl', 'rb') as file:
    #     data = pickle.load(file)
    # net = data['net']

    # for connection in net.connections:
    #     foo = int(round(connection.w))
    #     if foo % 2 == 0:
    #         foo += 1
    #     print('{:.4}->{}'.format(connection.w, foo))

    # c_scale = .75
    # sigma_scale = .75
    # subsample_indices = preprocess(net, c_scale, sigma_scale, 'PITv_2/3')

    # backbone = Backbone(net, 'PITv_2/3')
    # classifier = Classifier(backbone)
    # classifier.snip()
    # torch.save(classifier, 'classifier.pkl')
    # foo = classifier.forward(torch.Tensor(np.random.randn(8, 3, 32, 32)))
    # print(foo.shape)

    # backbone.forward(torch.Tensor(np.random.randn(8, 3, 32, 32)))

    # net = torch.load('/floyd/input/saved_models/classifier.pkl')

    if False:
        with open('optimization-result-PITv.pkl', 'rb') as file:
                data = pickle.load(file)
        net = data['net']
        backbone = Backbone(net, 'PITv_2/3', c_scale=.5, sigma_scale=.5)
    else:
        with open('optimization-result-AITv.pkl', 'rb') as file:
                data = pickle.load(file)
        net = data['net']
        backbone = Backbone(net, 'AITv_2/3', c_scale=.75, sigma_scale=.75)

    net = Classifier(backbone)
    net.snip(batches=1)

    checkpoint = torch.load('./checkpoint/ckpt.pth', map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    backbone.print()

    # from cifar import get_testloader, test
    # testloader = get_testloader()
    # criterion = nn.CrossEntropyLoss()
    # acc = test(net, testloader, criterion)

    # activities, categories = get_activities(
    #     './checkpoint/ckpt.pth',
    #     '../it-cnn/tuning/images/clutter',
    #     result_file='results/clutter-result.pkl')
    # print(activities)
    # print(categories)
