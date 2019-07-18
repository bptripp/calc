# CALC
CALC is software for generating convolutional network architectures that closely resemble the architecture of the primate visual cortex. Each layer in a CALC architecture corresponds with a cell population in the brain, specifically with the group of excitatory cells in a certain layer of a certain cortical area. Hyperparameters are optimized to match primate tract tracing data, cortical area sizes and cell densities, neuron-level in-degrees, and classical receptive field sizes where available. CALC is described in [this paper](https://www.mitpressjournals.org/doi/full/10.1162/neco_a_01211).

CALC was used to develop a convolutional architecture that matches a single hemisphere of the macaque monkey visual cortex, which we call the macaque single-hemisphere (MSH-CALC) model. Connections in this model are summarized in the diagram below. The colour indicates density of the connection. The density measure is the log-FLNe, or fraction of labelled neurons external to the injection site. This is a measure from retrograde tract-tracing studies that was used (among other data) to parameterize the architecture. Some labels are omitted to make the rest of the labels legible, but the model includes separate convolutional layers for cortical layers 2/3, 4, 5, and 6 in each visual area. 

![network diagram](https://github.com/bptripp/calc/blob/master/MSH-FLNe.png "network diagram")

# Pretrained Networks
Parts of the ventral stream have been trained on ImageNet. Here is a PyTorch network that includes LGN, V1, V2, V4, and PIT. It has top-1 accuracy of ~60% on ImageNet. [Download.](https://www.floydhub.com/bptripp/datasets/calc-trained/2/CALC_MSH_01a.zip) To load the model, unzip the file and run cnn_pytorch in Python 3 from the same directory. You need pytorch and numpy. 

# Sparsity 
Many of the connections in CALC networks are quite sparse, in keeping with estimates of connection sparsity in the brain. Interestingly this impairs performance. There are two sparsity parameters, an element-wise one and a channel-wise one. The graph below shows how CIFAR-10 performance is affected by sparsity. For this plot, the logarithm of each sparsity parameter was scaled by various factors. Physiological sparsity corresponds to both scale factors being 1, which gives the worst performance. As an aside, the best CIFAR-10 test accuracy obtained with this network was about 85% with further training. It isn't clear yet whether there is a better way to initialize or train the networks, or whether this points to a fundamental difference between communication in convolutional networks and the brain (e.g. perhaps the brain uses predictive coding, or carries some information in spike times). 

![sparsity](https://github.com/bptripp/calc/blob/master/sparsity.png "sparsity")
