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

# Concepts

## Interactions

  - An `environment` (traditionally made using OpenAI Gym, or using their `tf_agents.environments` options) returns a `time_step`.
    - That `time_step` can contain an array of numbers (the `observation`), the `reward` (if an action was taken), and whether or not it is the final `time_step` in an episode.

  - An `agent` (which includes a handful of configurable things inside it), will interact with this `environment` by reading its `time_step`, and deciding which `action` to take next.

  - That `action` is then sent into the `environment`, which of course then returns a brand new `time_step`.

## Agents

The `agent` can choose each `action` to either:

  1. increase the likelyhood of getting the most rewards from the `time_step` (using the `agent.policy`)
  2. or, it will choose actions to simply acquire data about the evironment (using the `agent.collect_policy`).

While collecting data from the environment, the `agent` will store three things at each step in a dataset. They are the current `time_step`, the `action` it chose, and then the resulting `time_step`. These can then be used in batches later one, to train the `agent.policy` (similar to training a traditional NN).

There are then implementation differences between the different algorithms, and many points at which the `agent` is configurable, all of which it would seem require a deeper knowledge of the tools and what you are trying to Learn.

## Data Types

### Observation

A numpy array (or tensor or nest) containing the input data of a policy network (also it is contained within the output from an environment).

### Action

A numpy array (or tensor or nest) containing the output data of a policy network (also it then fed to an environment to act upon).

### Policy State

A numpy array (or tensor or nest) containing a policy's previous state.

### Discount

A numpy array (or tensor or nest) with a range of [0, 1].
