ud985
=====

This repo contains a neural network implementation that
attempts to predict the number of clusters a given
community detection algorithm will generate on a network.
This is done by feeding the neural network a few pieces of
information about the network


Requirements
------------

* ndex2 client > 3.3.1 & < 4.0.0
* pandas
* pytorch
* networkx > 2.3
* matplotlib


Scripts and what they do
------------------------

* ``create_trainingdata.py``

  Takes networks created from ``generate_subgraph.py`` or ``generate_gosubgraph.py``
  along with output from a community detection algorithm and generates training
  data usable by ``predict_clusters.py``

* ``generate_subgraph.py``

  Creates networks from an input CX network
  by randomly picking nodes

* ``generate_gosubgraph.py``

  Creates networks from an input CX network
  by selecting nodes matching genes for GO terms

* ``predict_clusters.py``

  Runs training and prediction

*