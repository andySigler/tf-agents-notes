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

# `tf_agents.specs`

Specs allow classes to instatiate based on their expected input/output. Specs are used to document a numpy array or tensor's data type, the shape, and (optionally) the allowed ranges of a numpy array or tensor.

There are also some helper function within the submodules that are not listed here.

## Types

  - `ArraySpec`
    - `BoundedArraySpec` specifies a min and max
  - `DistributionSpec`
  - `TensorSpec`

## Arguments

  - `shape`
  - `dtype`
  - `name` (optional)

## Methods

  - `check_array(array)`
  - `from_array(array)`
  - `from_spec(spec)`
