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

# `tf_agents.drivers`

Helper functions, for running a `policy` within an `environment`.

## Arguments

Drivers use the base class `tf_agents.drivers.driver.Driver`, and take (mostly) three arguments:

  1. `environment`
  2. `policy`
  3. `observers`
    - this is an array of (optional) callback functions, which will take a `trajectory` as the argument

## Types

Currently there are three `Driver` subclasses:

  - `DynamicEpisodeDriver`
    - stops after certain number of episodes
  - `DynamicStepDriver`
    - stops after certain number of steps
  - `PyDriver`
    - runs a Python `environment` (not a TF `environment`)
