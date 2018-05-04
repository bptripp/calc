import keras
from keras.layers import Conv2D, Activation, BatchNormalization

def make_model_from_network(net, input, output_name):
    """
    Note the "output" layer returned from this method may not be the model's final output,
    but rather the nearest layer to the output with a direct physiological homologue. You
    can add layers after it to suit a certain task. For example, for CIFAR-10 you might have:

    output_layer = make_model_from_network(net, input, 'TEpd_5')
    layer_f = Flatten()(output_layer)
    layer_d1 = Dense(128, activation='relu')(layer_f)
    layer_d2 = Dense(128, activation='relu')(layer_d1)
    layer_classifier = Dense(10, activation='softmax')(layer_d2)
    model = keras.Model(inputs=input, outputs=layer_classifier)

    :param input: layer that provides images
    :param output_name: name of layer to use as output
    :return: output_layer: output layer (other layers can be added after this; see above)
    """

    input_name = 'INPUT'
    complete_layers = {input_name: input}

    print(len(complete_layers))
    print(complete_layers)

    while len(complete_layers) < len(net.layers):
        for layer in net.layers:
            if layer.name not in complete_layers.keys():
                # add this layer if all its inputs are there already
                inbounds = net.find_inbounds(layer.name)

                all_there = True
                for inbound in inbounds:
                    if inbound.pre.name not in complete_layers.keys():
                        all_there = False
                        break

                # TODO: Lambda for subsampling
                if all_there:
                    conv_layers = [] #one for each input
                    for inbound in inbounds:
                        m = int(layer.m)
                        w = int(inbound.w)
                        s = int(inbound.s)
                        print('origin: {} termination: {} m: {} w: {} stride: {}'.format(inbound.pre.name, layer.name, m, w, s))
                        name = '{}-{}'.format(inbound.pre.name, layer.name)
                        input_layer = complete_layers[inbound.pre.name]
                        conv_layer = Conv2D(m, (w, w), strides=(s, s), padding='same', name=name)(input_layer)
                        conv_layers.append(conv_layer)

                    if len(conv_layers) > 1:
                        print('adding converging paths')
                        x = keras.layers.add(conv_layers)
                    else:
                        x = conv_layers[0]

                    x = Activation('relu')(x)
                    x = BatchNormalization()(x)
                    complete_layers[layer.name] = x

                    print("adding " + layer.name)

    return complete_layers[output_name]
