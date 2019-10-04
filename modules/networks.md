# TF-Agents Notes

  - [Concepts](./concepts.md)
    - [Interactions](./concepts.md#Interactions)
    - [Agents](./concepts.md#Agents)
    - [Data Types](./concepts.md#data-types)
  - [Modules](./modules.md)
    - [tf_agents.specs](./tfagents_specs.md)
    - [tf_agents.trajectories](./tfagents_trajectories.md)
    - [tf_agents.replay_buffers](./tfagents_replay_buffers.md)
    - [tf_agents.environments](./tfagents_environments.md)
    - [tf_agents.agents](./tfagents_agents.md)
    - [tf_agents.drivers](./tfagents_drivers.md)
    - [tf_agents.policies](./tfagents_policies.md)
    - [tf_agents.networks](./tfagents_networks.md)
    - [tf_agents.metrics](./tfagents_metrics.md)
    - [tf_agents.utils](./tfagents_utils.md)
    - [tf_agents.distributions](./tfagents_distributions.md)

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
