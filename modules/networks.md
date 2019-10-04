# `tf_agents.networks`

The base classes are:

  - `tf_agents.networks.network.Network`
  - `tf_agents.networks.network.DistributionNetwork`

These seem to have similar methods to Keras networks, like save/load, layers, shapes, trainable, etc.

## Keras Layers

  - `tf_agents.networks.bias_layer`
  - `tf_agents.networks.expand_dims_layer`
  - `tf_agents.networks.sequential_layer`

## Keras Networks

  - `tf_agents.networks.encoding_network`
  - `tf_agents.networks.lstm_encoding_network`

## Keras Sample Networks

  - `tf_agents.networks.value_network`
  - `tf_agents.networks.value_rnn_network`
  - `tf_agents.networks.q_network`
  - `tf_agents.networks.q_rnn_network`
  - `tf_agents.networks.actor_distribution_network`
  - `tf_agents.networks.actor_distribution_rnn_network`

## Projections

Return a distribution from which you can sample from

  - `tf_agents.networks.categorical_projection_network`
  - `tf_agents.networks.normal_projection_network`
