# CALC
CALC is software for generating convolutional network architectures that closely resemble the architecture of the primate cortex. Each layer in a CALC architecture corresponds with a cell population in the brain, specifically with the group of pyramidal cells in a certain layer of a certain cortical area. Hyperparameters are optimized to match primate tract tracing data, cortical area sizes and cell densities, neuron-level in-degrees, and classical receptive field sizes where available. 

It is a work in progress.

## Source Data
In addition to the code, you will also need a few data files that are not redistributed here. Register on core-nets.org, and download the following files:
JCN_2013 Table.xls
Cercor_2012 Table.xls
Open each of these files, and save the first sheet in .csv format, in the calc/data/markov folder.
Then run check_data() in data.py to make sure you have these in the expected form.
